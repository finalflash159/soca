from pathlib import Path

import sounddevice as sd
import soundfile as sf

OUT_DIR = Path(__file__).parent.parent / "data" / "test_audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("sample_01_greeting", "Xin chào, tôi là Soca.", 3),
    ("sample_02_question", "Hôm nay thời tiết Sài Gòn thế nào?", 4),
    ("sample_03_command", "Đặt báo thức lúc bảy giờ sáng mai.", 4),
]

SAMPLE_RATE = 16000
print("Available input devices:")
print(sd.query_devices())
print()

for name, prompt, duration in PROMPTS:
    out_path = OUT_DIR / f"{name}.wav"
    if out_path.exists():
        print(f"⏭️  Skipping {name} (already exists)")
        continue
    input(f"\n>>> Press ENTER, then say: \"{prompt}\" ({duration}s)...")
    print(f"🎙️  Recording {duration}s...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(str(out_path), audio, SAMPLE_RATE)
    print(f"✅ Saved {out_path}")

print(f"\n🎉 Done. Files in: {OUT_DIR}")
