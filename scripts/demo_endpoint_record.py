from pathlib import Path

import soundfile as sf
from rich.console import Console

from soca.asr import SpeechDetector
from soca.core import EndpointConfig, record_until_silence

console = Console()

config = EndpointConfig()

console.print(
    "[bold green]Recording...[/bold green] Speak into your microphone. "
    f"Recording will stop after {config.endpoint_silence_ms} ms of silence."
)
detector = SpeechDetector()
audio = record_until_silence(detector, config=config)

out = Path("/tmp/soca_endpoint.wav")
sf.write(out, audio, config.sample_rate)

console.print(f"Saved: {out}")
console.print(f"Duration: {len(audio) / config.sample_rate:.2f}s")
