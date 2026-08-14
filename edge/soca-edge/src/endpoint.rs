use serde::Serialize;
use thiserror::Error;

pub const VAD_FRAME_SAMPLES: usize = 512;
pub const VAD_FRAME_MS: u32 = 32;

pub trait VoiceActivityDetector {
    fn probability(&mut self, frame: &[f32]) -> Result<f32, InferenceError>;
    fn reset(&mut self) -> Result<(), InferenceError>;
}

pub trait TurnCompletionDetector {
    fn probability_complete(&mut self, audio: &[f32]) -> Result<f32, InferenceError>;
}

#[derive(Debug, Error)]
#[error("{model} inference failed: {detail}")]
pub struct InferenceError {
    pub model: &'static str,
    pub detail: String,
}

#[derive(Clone, Copy, Debug)]
pub struct EndpointConfig {
    pub vad_threshold: f32,
    pub floor_silence_ms: u32,
    pub ceil_silence_ms: u32,
    pub max_turn_ms: u32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.5,
            floor_silence_ms: 1_800,
            ceil_silence_ms: 3_000,
            max_turn_ms: 30_000,
        }
    }
}

impl EndpointConfig {
    pub fn validate(self) -> Result<Self, EndpointError> {
        if !(0.0..=1.0).contains(&self.vad_threshold) {
            return Err(EndpointError::InvalidConfig("vad_threshold"));
        }
        if self.floor_silence_ms == 0
            || self.ceil_silence_ms < self.floor_silence_ms
            || self.max_turn_ms <= self.ceil_silence_ms
        {
            return Err(EndpointError::InvalidConfig("silence_or_turn_bounds"));
        }
        Ok(self)
    }
}

#[derive(Debug, Error)]
pub enum EndpointError {
    #[error("invalid endpoint config: {0}")]
    InvalidConfig(&'static str),
    #[error(transparent)]
    Inference(#[from] InferenceError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointReason {
    AdaptiveSilence,
    MaximumDuration,
}

#[derive(Debug, Serialize)]
pub struct CompletedTurn {
    pub reason: EndpointReason,
    pub audio: Vec<f32>,
    pub duration_ms: u32,
    pub required_silence_ms: u32,
    pub speech_frames: u32,
}

pub struct EndpointController<V, T> {
    vad: V,
    turn: T,
    config: EndpointConfig,
    audio: Vec<f32>,
    frame: Vec<f32>,
    speech_started: bool,
    silence_ms: u32,
    speech_frames: u32,
}

impl<V, T> EndpointController<V, T>
where
    V: VoiceActivityDetector,
    T: TurnCompletionDetector,
{
    pub fn new(vad: V, turn: T, config: EndpointConfig) -> Result<Self, EndpointError> {
        Ok(Self {
            vad,
            turn,
            config: config.validate()?,
            audio: Vec::new(),
            frame: Vec::with_capacity(VAD_FRAME_SAMPLES),
            speech_started: false,
            silence_ms: 0,
            speech_frames: 0,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<CompletedTurn>, EndpointError> {
        let mut completed = Vec::new();
        for &sample in samples {
            self.frame.push(sample.clamp(-1.0, 1.0));
            if self.frame.len() == VAD_FRAME_SAMPLES
                && let Some(turn) = self.process_frame()?
            {
                completed.push(turn);
            }
        }
        Ok(completed)
    }

    fn process_frame(&mut self) -> Result<Option<CompletedTurn>, EndpointError> {
        let probability = self.vad.probability(&self.frame)?;
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(InferenceError {
                model: "silero_vad",
                detail: "probability_out_of_range".to_owned(),
            }
            .into());
        }
        self.audio.extend_from_slice(&self.frame);
        self.frame.clear();
        if probability >= self.config.vad_threshold {
            self.speech_started = true;
            self.speech_frames += 1;
            self.silence_ms = 0;
        } else if self.speech_started {
            self.silence_ms = self.silence_ms.saturating_add(VAD_FRAME_MS);
        }
        let duration_ms =
            u32::try_from(self.audio.len().saturating_mul(1_000) / 16_000).unwrap_or(u32::MAX);
        if duration_ms >= self.config.max_turn_ms {
            return self.finish(EndpointReason::MaximumDuration, self.config.ceil_silence_ms);
        }
        if !self.speech_started || self.silence_ms < self.config.floor_silence_ms {
            return Ok(None);
        }
        let p_complete = self.turn.probability_complete(&self.audio)?;
        if !p_complete.is_finite() || !(0.0..=1.0).contains(&p_complete) {
            return Err(InferenceError {
                model: "smart_turn",
                detail: "probability_out_of_range".to_owned(),
            }
            .into());
        }
        let span = self.config.ceil_silence_ms - self.config.floor_silence_ms;
        let required =
            self.config.floor_silence_ms + ((span as f32) * (1.0 - p_complete)).round() as u32;
        if self.silence_ms < required {
            return Ok(None);
        }
        self.finish(EndpointReason::AdaptiveSilence, required)
    }

    fn finish(
        &mut self,
        reason: EndpointReason,
        required_silence_ms: u32,
    ) -> Result<Option<CompletedTurn>, EndpointError> {
        let audio = std::mem::take(&mut self.audio);
        let duration_ms =
            u32::try_from(audio.len().saturating_mul(1_000) / 16_000).unwrap_or(u32::MAX);
        let speech_frames = self.speech_frames;
        self.frame.clear();
        self.speech_started = false;
        self.silence_ms = 0;
        self.speech_frames = 0;
        self.vad.reset()?;
        Ok(Some(CompletedTurn {
            reason,
            audio,
            duration_ms,
            required_silence_ms,
            speech_frames,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnergyVad;
    impl VoiceActivityDetector for EnergyVad {
        fn probability(&mut self, frame: &[f32]) -> Result<f32, InferenceError> {
            Ok(if frame.iter().any(|sample| sample.abs() > 0.1) {
                1.0
            } else {
                0.0
            })
        }

        fn reset(&mut self) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    struct Complete(f32);
    impl TurnCompletionDetector for Complete {
        fn probability_complete(&mut self, _audio: &[f32]) -> Result<f32, InferenceError> {
            Ok(self.0)
        }
    }

    fn frames(value: f32, count: usize) -> Vec<f32> {
        vec![value; VAD_FRAME_SAMPLES * count]
    }

    #[test]
    fn complete_turn_stops_at_floor_without_cutting_speech() {
        let config = EndpointConfig {
            floor_silence_ms: 64,
            ceil_silence_ms: 128,
            max_turn_ms: 2_000,
            ..EndpointConfig::default()
        };
        let mut endpoint = EndpointController::new(EnergyVad, Complete(1.0), config).unwrap();
        assert!(endpoint.push(&frames(0.5, 2)).unwrap().is_empty());
        let completed = endpoint.push(&frames(0.0, 2)).unwrap();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].required_silence_ms, 64);
        assert_eq!(completed[0].reason, EndpointReason::AdaptiveSilence);
    }

    #[test]
    fn incomplete_turn_holds_until_ceiling() {
        let config = EndpointConfig {
            floor_silence_ms: 64,
            ceil_silence_ms: 128,
            max_turn_ms: 2_000,
            ..EndpointConfig::default()
        };
        let mut endpoint = EndpointController::new(EnergyVad, Complete(0.0), config).unwrap();
        endpoint.push(&frames(0.5, 2)).unwrap();
        assert!(endpoint.push(&frames(0.0, 3)).unwrap().is_empty());
        let completed = endpoint.push(&frames(0.0, 1)).unwrap();
        assert_eq!(completed[0].required_silence_ms, 128);
    }
}
