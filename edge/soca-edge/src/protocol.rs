use serde::Serialize;

use crate::endpoint::EndpointReason;

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EdgeEvent<'a> {
    Ready {
        protocol: &'static str,
        device: &'a str,
        source_sample_rate: u32,
        target_sample_rate: u32,
        silero_sha256: &'a str,
        smart_turn_sha256: &'a str,
    },
    Turn {
        sequence: u64,
        terminal: &'static str,
        reason: EndpointReason,
        sample_rate: u32,
        duration_ms: u32,
        required_silence_ms: u32,
        speech_frames: u32,
        audio_encoding: &'static str,
        audio_base64: &'a str,
    },
    Failure {
        code: &'a str,
        detail: &'a str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_is_one_typed_ndjson_object() {
        let payload = serde_json::to_string(&EdgeEvent::Turn {
            sequence: 1,
            terminal: "endpoint",
            reason: EndpointReason::AdaptiveSilence,
            sample_rate: 16_000,
            duration_ms: 900,
            required_silence_ms: 1_800,
            speech_frames: 20,
            audio_encoding: "pcm_s16le",
            audio_base64: "AAAA",
        })
        .unwrap();
        assert!(!payload.contains('\n'));
        assert!(payload.contains("\"type\":\"turn\""));
        assert!(payload.contains("\"terminal\":\"endpoint\""));
    }
}
