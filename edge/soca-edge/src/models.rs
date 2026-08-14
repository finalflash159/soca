use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

use ort::session::Session;
use ort::value::Tensor;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use crate::endpoint::{InferenceError, TurnCompletionDetector, VoiceActivityDetector};

const SILERO_CONTEXT: usize = 64;
const SILERO_FRAME: usize = 512;
const SILERO_STATE: usize = 2 * 128;
const SMART_TURN_SAMPLES: usize = 8 * 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const MEL_BINS: usize = 80;
const MEL_FRAMES: usize = 800;

fn inference_error(model: &'static str, error: impl std::fmt::Display) -> InferenceError {
    InferenceError {
        model,
        detail: error.to_string(),
    }
}

fn build_session(path: &Path, model: &'static str) -> Result<Session, InferenceError> {
    if !path.is_file() {
        return Err(inference_error(
            model,
            format!("missing model: {}", path.display()),
        ));
    }
    let builder = Session::builder().map_err(|error| inference_error(model, error))?;
    let builder = builder
        .with_intra_threads(1)
        .map_err(|error| inference_error(model, error))?;
    let mut builder = builder
        .with_inter_threads(1)
        .map_err(|error| inference_error(model, error))?;
    builder
        .commit_from_file(path)
        .map_err(|error| inference_error(model, error))
}

pub struct OrtSileroVad {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
}

impl OrtSileroVad {
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        Ok(Self {
            session: build_session(path, "silero_vad")?,
            state: vec![0.0; SILERO_STATE],
            context: vec![0.0; SILERO_CONTEXT],
        })
    }
}

impl VoiceActivityDetector for OrtSileroVad {
    fn probability(&mut self, frame: &[f32]) -> Result<f32, InferenceError> {
        if frame.len() != SILERO_FRAME {
            return Err(inference_error(
                "silero_vad",
                "expected exactly 512 samples",
            ));
        }
        let mut input = Vec::with_capacity(SILERO_CONTEXT + SILERO_FRAME);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(frame);
        self.context
            .copy_from_slice(&input[input.len() - SILERO_CONTEXT..]);
        let input = Tensor::from_array(([1, SILERO_CONTEXT + SILERO_FRAME], input))
            .map_err(|error| inference_error("silero_vad", error))?;
        let state = Tensor::from_array(([2, 1, 128], self.state.clone()))
            .map_err(|error| inference_error("silero_vad", error))?;
        let sample_rate = Tensor::from_array(([] as [usize; 0], vec![16_000_i64]))
            .map_err(|error| inference_error("silero_vad", error))?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input,
                "state" => state,
                "sr" => sample_rate,
            ])
            .map_err(|error| inference_error("silero_vad", error))?;
        let (_, probabilities) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|error| inference_error("silero_vad", error))?;
        let (_, next_state) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|error| inference_error("silero_vad", error))?;
        if probabilities.len() != 1 || next_state.len() != SILERO_STATE {
            return Err(inference_error("silero_vad", "unexpected output shape"));
        }
        self.state.copy_from_slice(next_state);
        Ok(probabilities[0].clamp(0.0, 1.0))
    }

    fn reset(&mut self) -> Result<(), InferenceError> {
        self.state.fill(0.0);
        self.context.fill(0.0);
        Ok(())
    }
}

pub struct OrtSmartTurn {
    session: Session,
    feature_extractor: WhisperFeatures,
}

impl OrtSmartTurn {
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        Ok(Self {
            session: build_session(path, "smart_turn")?,
            feature_extractor: WhisperFeatures::new(),
        })
    }
}

impl TurnCompletionDetector for OrtSmartTurn {
    fn probability_complete(&mut self, audio: &[f32]) -> Result<f32, InferenceError> {
        let features = self.feature_extractor.extract(audio);
        let input = Tensor::from_array(([1, MEL_BINS, MEL_FRAMES], features))
            .map_err(|error| inference_error("smart_turn", error))?;
        let outputs = self
            .session
            .run(ort::inputs!["input_features" => input])
            .map_err(|error| inference_error("smart_turn", error))?;
        let (_, probabilities) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|error| inference_error("smart_turn", error))?;
        if probabilities.len() != 1 {
            return Err(inference_error("smart_turn", "unexpected output shape"));
        }
        Ok(probabilities[0].clamp(0.0, 1.0))
    }
}

struct WhisperFeatures {
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    filters: Vec<f32>,
}

impl WhisperFeatures {
    fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let window = (0..N_FFT)
            .map(|index| 0.5 - 0.5 * ((2.0 * PI * index as f32) / N_FFT as f32).cos())
            .collect();
        Self {
            fft,
            window,
            filters: mel_filter_bank(),
        }
    }

    fn extract(&self, audio: &[f32]) -> Vec<f32> {
        let mut waveform = vec![0.0; SMART_TURN_SAMPLES];
        if audio.len() >= SMART_TURN_SAMPLES {
            waveform.copy_from_slice(&audio[audio.len() - SMART_TURN_SAMPLES..]);
        } else {
            waveform[SMART_TURN_SAMPLES - audio.len()..].copy_from_slice(audio);
        }
        normalize(&mut waveform);
        let padded = reflect_pad(&waveform, N_FFT / 2);
        let mut features = vec![0.0_f32; MEL_BINS * MEL_FRAMES];
        let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); N_FFT];
        for frame_index in 0..MEL_FRAMES {
            let start = frame_index * HOP;
            for (index, value) in spectrum.iter_mut().enumerate() {
                *value = Complex::new(padded[start + index] * self.window[index], 0.0);
            }
            self.fft.process(&mut spectrum);
            for mel in 0..MEL_BINS {
                let mut energy = 0.0_f32;
                for (frequency, value) in spectrum.iter().take(N_FFT / 2 + 1).enumerate() {
                    energy += value.norm_sqr() * self.filters[frequency * MEL_BINS + mel];
                }
                features[mel * MEL_FRAMES + frame_index] = energy.max(1.0e-10).log10();
            }
        }
        let maximum = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut features {
            *value = (value.max(maximum - 8.0) + 4.0) / 4.0;
        }
        features
    }
}

fn normalize(waveform: &mut [f32]) {
    let mean = waveform.iter().map(|&value| f64::from(value)).sum::<f64>() / waveform.len() as f64;
    let variance = waveform
        .iter()
        .map(|&value| {
            let centered = f64::from(value) - mean;
            centered * centered
        })
        .sum::<f64>()
        / waveform.len() as f64;
    let scale = (variance + 1.0e-7).sqrt();
    for value in waveform {
        *value = ((f64::from(*value) - mean) / scale) as f32;
    }
}

fn reflect_pad(waveform: &[f32], padding: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(waveform.len() + 2 * padding);
    padded.extend((1..=padding).rev().map(|index| waveform[index]));
    padded.extend_from_slice(waveform);
    padded.extend((1..=padding).map(|index| waveform[waveform.len() - 1 - index]));
    padded
}

fn hertz_to_mel(frequency: f32) -> f32 {
    if frequency < 1_000.0 {
        3.0 * frequency / 200.0
    } else {
        15.0 + (frequency / 1_000.0).ln() * (27.0 / 6.4_f32.ln())
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    if mel < 15.0 {
        200.0 * mel / 3.0
    } else {
        1_000.0 * ((6.4_f32.ln() / 27.0) * (mel - 15.0)).exp()
    }
}

fn mel_filter_bank() -> Vec<f32> {
    let min_mel = hertz_to_mel(0.0);
    let max_mel = hertz_to_mel(8_000.0);
    let centers: Vec<f32> = (0..MEL_BINS + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f32 / (MEL_BINS + 1) as f32;
            mel_to_hertz(mel)
        })
        .collect();
    let mut filters = vec![0.0_f32; (N_FFT / 2 + 1) * MEL_BINS];
    for frequency in 0..=N_FFT / 2 {
        let hertz = 8_000.0 * frequency as f32 / (N_FFT / 2) as f32;
        for mel in 0..MEL_BINS {
            let down = (hertz - centers[mel]) / (centers[mel + 1] - centers[mel]);
            let up = (centers[mel + 2] - hertz) / (centers[mel + 2] - centers[mel + 1]);
            let triangle = down.min(up).max(0.0);
            let normalization = 2.0 / (centers[mel + 2] - centers[mel]);
            filters[frequency * MEL_BINS + mel] = triangle * normalization;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn silence_matches_whisper_feature_floor() {
        let features = WhisperFeatures::new().extract(&vec![0.0; SMART_TURN_SAMPLES]);
        assert_eq!(features.len(), MEL_BINS * MEL_FRAMES);
        for value in features {
            assert_abs_diff_eq!(value, -1.5, epsilon = 1.0e-5);
        }
    }

    #[test]
    fn feature_extractor_is_finite_for_a_tone() {
        let audio: Vec<f32> = (0..SMART_TURN_SAMPLES)
            .map(|sample| (2.0 * PI * 440.0 * sample as f32 / 16_000.0).sin())
            .collect();
        let features = WhisperFeatures::new().extract(&audio);
        assert!(features.iter().all(|value| value.is_finite()));
        assert_abs_diff_eq!(features[0], 1.209, epsilon = 0.02);
        assert_abs_diff_eq!(features[10 * MEL_FRAMES + 100], 1.574, epsilon = 0.02);
    }
}
