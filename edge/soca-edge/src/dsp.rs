use thiserror::Error;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResampleError {
    #[error("sample rates must be positive")]
    InvalidSampleRate,
}

/// Stateful linear resampler for the capture bridge.
///
/// It intentionally runs outside the audio callback. The callback only writes
/// mono samples into the bounded SPSC ring; resampling and inference happen on
/// the consumer thread.
#[derive(Debug)]
pub struct LinearResampler {
    source_rate: u32,
    target_rate: u32,
    source_position: f64,
    previous: Option<f32>,
}

impl LinearResampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Result<Self, ResampleError> {
        if source_rate == 0 || target_rate == 0 {
            return Err(ResampleError::InvalidSampleRate);
        }
        Ok(Self {
            source_rate,
            target_rate,
            source_position: 0.0,
            previous: None,
        })
    }

    #[must_use]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if self.source_rate == self.target_rate {
            self.previous = input.last().copied();
            return input.to_vec();
        }

        let mut samples = Vec::with_capacity(input.len() + 1);
        if let Some(previous) = self.previous {
            samples.push(previous);
        }
        samples.extend_from_slice(input);
        self.previous = input.last().copied();

        let step = f64::from(self.source_rate) / f64::from(self.target_rate);
        let mut output = Vec::new();
        while self.source_position + 1.0 < samples.len() as f64 {
            let left = self.source_position.floor() as usize;
            let fraction = (self.source_position - left as f64) as f32;
            output.push(samples[left].mul_add(1.0 - fraction, samples[left + 1] * fraction));
            self.source_position += step;
        }
        self.source_position -= (samples.len() - 1) as f64;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_preserves_samples() {
        let mut resampler = LinearResampler::new(16_000, 16_000).unwrap();
        assert_eq!(resampler.process(&[0.0, 0.5, -0.5]), [0.0, 0.5, -0.5]);
    }

    #[test]
    fn forty_eight_khz_downsamples_to_sixteen_khz_across_chunks() {
        let mut resampler = LinearResampler::new(48_000, 16_000).unwrap();
        let first = resampler.process(&[0.0, 1.0, 2.0, 3.0]);
        let second = resampler.process(&[4.0, 5.0, 6.0]);
        assert_eq!(first, [0.0]);
        assert_eq!(second, [3.0]);
    }
}
