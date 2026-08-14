use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use rtrb::{Producer, RingBuffer};
use serde::Serialize;
use sha2::{Digest, Sha256};
use soca_edge::dsp::{LinearResampler, TARGET_SAMPLE_RATE};
use soca_edge::endpoint::{EndpointConfig, EndpointController};
use soca_edge::models::{OrtSileroVad, OrtSmartTurn};
use soca_edge::protocol::EdgeEvent;

const PROTOCOL: &str = "soca-edge-ndjson-v1";
const RING_SECONDS: usize = 10;
const CONSUMER_CHUNK: usize = 2_048;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(long)]
    silero_model: PathBuf,
    #[arg(long)]
    smart_turn_model: PathBuf,
    #[arg(long)]
    device: Option<String>,
    #[arg(long, default_value_t = 0.5)]
    vad_threshold: f32,
    #[arg(long, default_value_t = 1_800)]
    floor_silence_ms: u32,
    #[arg(long, default_value_t = 3_000)]
    ceil_silence_ms: u32,
    #[arg(long, default_value_t = 30_000)]
    max_turn_ms: u32,
    /// Exit after this duration; required for a reproducible device receipt.
    #[arg(long, default_value_t = 0)]
    measurement_seconds: u64,
    #[arg(long, requires = "measurement_seconds")]
    receipt: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct DeviceReceipt {
    schema_version: &'static str,
    gate_target: &'static str,
    os: &'static str,
    architecture: &'static str,
    device: String,
    source_sample_rate: u32,
    target_sample_rate: u32,
    capture_seconds: f64,
    completed_turns: u64,
    dropped_capture_samples: u64,
    stream_error: bool,
    processing_latency_p95_ms: f64,
    peak_rss_kib: Option<u64>,
    silero_sha256: String,
    smart_turn_sha256: String,
    protocol: &'static str,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let endpoint_config = EndpointConfig {
        vad_threshold: args.vad_threshold,
        floor_silence_ms: args.floor_silence_ms,
        ceil_silence_ms: args.ceil_silence_ms,
        max_turn_ms: args.max_turn_ms,
    }
    .validate()
    .context("invalid endpoint configuration")?;
    let silero_sha256 = file_sha256(&args.silero_model)?;
    let smart_turn_sha256 = file_sha256(&args.smart_turn_model)?;
    let vad = OrtSileroVad::load(&args.silero_model).context("Silero startup failed")?;
    let turn = OrtSmartTurn::load(&args.smart_turn_model).context("Smart Turn startup failed")?;
    let mut endpoint = EndpointController::new(vad, turn, endpoint_config)?;

    let host = cpal::default_host();
    let device = select_device(&host, args.device.as_deref())?;
    let device_name = device.description().map_or_else(
        |_| "unknown-input-device".to_owned(),
        |description| description.name().to_owned(),
    );
    let supported = device
        .default_input_config()
        .context("default input configuration is unavailable")?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let source_rate = config.sample_rate;
    let channels = usize::from(config.channels);
    if channels == 0 {
        return Err(anyhow!("input device reported zero channels"));
    }

    let ring_capacity = usize::try_from(source_rate)
        .unwrap_or(48_000)
        .saturating_mul(RING_SECONDS);
    let (producer, mut consumer) = RingBuffer::<f32>::new(ring_capacity);
    let dropped = Arc::new(AtomicU64::new(0));
    let stream_error = Arc::new(AtomicBool::new(false));
    let stream = build_stream(
        &device,
        &config,
        sample_format,
        producer,
        Arc::clone(&dropped),
        Arc::clone(&stream_error),
        channels,
    )?;

    let stdout = std::io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    emit(
        &mut output,
        &EdgeEvent::Ready {
            protocol: PROTOCOL,
            device: &device_name,
            source_sample_rate: source_rate,
            target_sample_rate: TARGET_SAMPLE_RATE,
            silero_sha256: &silero_sha256,
            smart_turn_sha256: &smart_turn_sha256,
        },
    )?;
    stream.play().context("failed to start input stream")?;

    let started = Instant::now();
    let measurement =
        (args.measurement_seconds > 0).then(|| Duration::from_secs(args.measurement_seconds));
    let mut resampler = LinearResampler::new(source_rate, TARGET_SAMPLE_RATE)?;
    let mut capture_chunk = Vec::with_capacity(CONSUMER_CHUNK);
    let mut latencies_ms = Vec::new();
    let mut completed_turns = 0_u64;
    loop {
        if stream_error.load(Ordering::Relaxed) {
            emit(
                &mut output,
                &EdgeEvent::Failure {
                    code: "capture_stream_failed",
                    detail: "CPAL reported a terminal stream error",
                },
            )?;
            break;
        }
        capture_chunk.clear();
        while capture_chunk.len() < CONSUMER_CHUNK {
            match consumer.pop() {
                Ok(sample) => capture_chunk.push(sample),
                Err(_) => break,
            }
        }
        if capture_chunk.is_empty() {
            std::thread::sleep(Duration::from_millis(2));
        } else {
            let processing_started = Instant::now();
            let samples = resampler.process(&capture_chunk);
            for completed in endpoint.push(&samples)? {
                completed_turns += 1;
                let pcm = pcm_s16le(&completed.audio);
                let encoded = STANDARD.encode(pcm);
                emit(
                    &mut output,
                    &EdgeEvent::Turn {
                        sequence: completed_turns,
                        terminal: "endpoint",
                        reason: completed.reason,
                        sample_rate: TARGET_SAMPLE_RATE,
                        duration_ms: completed.duration_ms,
                        required_silence_ms: completed.required_silence_ms,
                        speech_frames: completed.speech_frames,
                        audio_encoding: "pcm_s16le",
                        audio_base64: &encoded,
                    },
                )?;
            }
            latencies_ms.push(processing_started.elapsed().as_secs_f64() * 1_000.0);
        }
        if measurement.is_some_and(|duration| started.elapsed() >= duration) {
            break;
        }
    }
    drop(stream);

    if let Some(path) = args.receipt {
        let receipt = DeviceReceipt {
            schema_version: "soca-edge-device-receipt-v1",
            gate_target: "linux_aarch64_sbc",
            os: std::env::consts::OS,
            architecture: std::env::consts::ARCH,
            device: device_name,
            source_sample_rate: source_rate,
            target_sample_rate: TARGET_SAMPLE_RATE,
            capture_seconds: started.elapsed().as_secs_f64(),
            completed_turns,
            dropped_capture_samples: dropped.load(Ordering::Relaxed),
            stream_error: stream_error.load(Ordering::Relaxed),
            processing_latency_p95_ms: percentile_95(&mut latencies_ms),
            peak_rss_kib: linux_peak_rss_kib(),
            silero_sha256,
            smart_turn_sha256,
            protocol: PROTOCOL,
        };
        let file = File::create(&path)
            .with_context(|| format!("cannot create receipt {}", path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), &receipt)?;
    }
    if stream_error.load(Ordering::Relaxed) {
        return Err(anyhow!("capture_stream_failed"));
    }
    Ok(())
}

fn select_device(host: &cpal::Host, requested: Option<&str>) -> Result<Device> {
    if let Some(name) = requested {
        return host
            .input_devices()
            .context("failed to enumerate input devices")?
            .find(|device| {
                device
                    .description()
                    .is_ok_and(|description| description.name() == name)
            })
            .ok_or_else(|| anyhow!("input device not found: {name}"));
    }
    host.default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))
}

fn build_stream(
    device: &Device,
    config: &StreamConfig,
    format: SampleFormat,
    producer: Producer<f32>,
    dropped: Arc<AtomicU64>,
    stream_error: Arc<AtomicBool>,
    channels: usize,
) -> Result<Stream> {
    let error_callback = move |error| {
        eprintln!("capture_stream_failed:{error}");
        stream_error.store(true, Ordering::Relaxed);
    };
    let stream = match format {
        SampleFormat::F32 => {
            input_stream_f32(device, config, producer, dropped, channels, error_callback)
        }
        SampleFormat::I16 => {
            input_stream_i16(device, config, producer, dropped, channels, error_callback)
        }
        SampleFormat::U16 => {
            input_stream_u16(device, config, producer, dropped, channels, error_callback)
        }
        unsupported => return Err(anyhow!("unsupported input sample format: {unsupported}")),
    }?;
    Ok(stream)
}

fn input_stream_f32<E>(
    device: &Device,
    config: &StreamConfig,
    mut producer: Producer<f32>,
    dropped: Arc<AtomicU64>,
    channels: usize,
    error_callback: E,
) -> Result<Stream, cpal::Error>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    device.build_input_stream(
        *config,
        move |data: &[f32], _| push_mono(data.iter().copied(), &mut producer, &dropped, channels),
        error_callback,
        None,
    )
}

fn input_stream_i16<E>(
    device: &Device,
    config: &StreamConfig,
    mut producer: Producer<f32>,
    dropped: Arc<AtomicU64>,
    channels: usize,
    error_callback: E,
) -> Result<Stream, cpal::Error>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    device.build_input_stream(
        *config,
        move |data: &[i16], _| {
            push_mono(
                data.iter()
                    .map(|&sample| f32::from(sample) / f32::from(i16::MAX)),
                &mut producer,
                &dropped,
                channels,
            );
        },
        error_callback,
        None,
    )
}

fn input_stream_u16<E>(
    device: &Device,
    config: &StreamConfig,
    mut producer: Producer<f32>,
    dropped: Arc<AtomicU64>,
    channels: usize,
    error_callback: E,
) -> Result<Stream, cpal::Error>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    device.build_input_stream(
        *config,
        move |data: &[u16], _| {
            push_mono(
                data.iter()
                    .map(|&sample| f32::from(sample) / 32_767.5 - 1.0),
                &mut producer,
                &dropped,
                channels,
            );
        },
        error_callback,
        None,
    )
}

fn push_mono(
    samples: impl Iterator<Item = f32>,
    producer: &mut Producer<f32>,
    dropped: &AtomicU64,
    channels: usize,
) {
    for sample in samples.step_by(channels) {
        if producer.push(sample).is_err() {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn emit(output: &mut impl Write, event: &EdgeEvent<'_>) -> Result<()> {
    serde_json::to_writer(&mut *output, event)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn pcm_s16le(audio: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(audio.len() * 2);
    for &sample in audio {
        let integer = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        bytes.extend_from_slice(&integer.to_le_bytes());
    }
    bytes
}

fn percentile_95(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index]
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn linux_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?.trim();
        value.split_whitespace().next()?.parse().ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_encoding_is_little_endian_and_bounded() {
        assert_eq!(pcm_s16le(&[-2.0, 0.0, 2.0]), [1, 128, 0, 0, 255, 127]);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let mut values = vec![3.0, 1.0, 2.0, 100.0];
        assert!((percentile_95(&mut values) - 100.0).abs() < f64::EPSILON);
    }
}
