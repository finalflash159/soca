"""Test Silero VAD on 3 conditions: real speech, silence, background noise."""

import sounddevice as sd
from rich.console import Console
from rich.table import Table

from soca.asr.vad import SpeechDetector

console = Console()
detector = SpeechDetector(threshold=0.5)
console.print("[green]✓ Silero VAD loaded[/green]\n")

SAMPLE_RATE = 16000
TEST_DURATION = 5

scenarios = [
    ("Speech test", "Nói rõ ràng: 'Mấy giờ rồi?'"),
    ("Silence test", "Hoàn toàn im lặng (đừng nói gì, đừng động đậy)"),
    ("Noise test", "Để mic gần quạt/AC/keyboard typing nhưng đừng nói"),
]

table = Table(title="Silero VAD — 3 Condition Test")
table.add_column("Scenario", style="cyan")
table.add_column("Has speech?", style="yellow")
table.add_column("Speech (ms)", justify="right")
table.add_column("Total (ms)", justify="right")
table.add_column("Ratio", justify="right", style="magenta")
table.add_column("# segments", justify="right")
table.add_column("VAD latency", justify="right", style="green")

for name, instruction in scenarios:
    console.print(f"\n[bold]{name}[/bold]: {instruction}")
    input(">>> Press ENTER to start recording...")
    console.print(f"[yellow]Recording {TEST_DURATION}s...[/yellow]")
    audio = sd.rec(
        int(TEST_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.flatten()

    result = detector.detect(audio)
    table.add_row(
        name,
        "✅ YES" if result.has_speech else "❌ NO",
        f"{result.speech_duration_ms:.0f}",
        f"{result.original_duration_ms:.0f}",
        f"{result.speech_ratio:.2%}",
        str(result.n_speech_segments),
        f"{result.vad_latency_ms:.0f} ms",
    )

console.print(table)
