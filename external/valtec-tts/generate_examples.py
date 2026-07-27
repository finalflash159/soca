"""Generate 5-speaker example audio files."""
import sys
sys.path.insert(0, '.')

from src.vietnamese.text_processor import process_vietnamese_text
from src.vietnamese.phonemizer import text_to_phonemes
from src.models.synthesizer import SynthesizerTrn
from src.text.symbols import symbols
import torch
import soundfile as sf
import json

# Load config
with open('pretrained/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SynthesizerTrn(
    len(symbols),
    config['data']['filter_length'] // 2 + 1,
    config['train']['segment_size'] // config['data']['hop_length'],
    n_speakers=config['data']['n_speakers'],
    **config['model'],
).to(device)

# Load checkpoint
checkpoint = torch.load('pretrained/G.pth', map_location=device)
state_dict = checkpoint['model']
new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Text to synthesize
text = "Tiếng xe cộ nhộn nhịp và ánh nắng len qua từng con phố nhỏ."
speakers = [('NF', 0), ('SF', 1), ('NM1', 2), ('SM', 3), ('NM2', 4)]

print(f"Text: {text}")
print(f"Generating examples for {len(speakers)} speakers...")

# Generate audio
for speaker_name, speaker_id in speakers:
    print(f"\nGenerating {speaker_name}...")
    
    # Prepare inputs
    from src.text import cleaned_text_to_sequence
    from src.nn import commons
    
    normalized_text = process_vietnamese_text(text)
    phones, tones, word2ph = text_to_phonemes(normalized_text, use_viphoneme=True)
    phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, "VI")
    
    # Add blanks
    phone_ids = commons.intersperse(phone_ids, 0)
    tone_ids = commons.intersperse(tone_ids, 0)
    lang_ids = commons.intersperse(lang_ids, 0)
    
    # Create tensors
    x = torch.LongTensor(phone_ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(phone_ids)]).to(device)
    tone = torch.LongTensor(tone_ids).unsqueeze(0).to(device)
    language = torch.LongTensor(lang_ids).unsqueeze(0).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)
    bert = torch.zeros(1, 1024, len(phone_ids)).to(device)
    ja_bert = torch.zeros(1, 768, len(phone_ids)).to(device)
    
    # Synthesize
    with torch.no_grad():
        audio, _, *_ = model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            sdp_ratio=0.0,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
        )
    
    audio = audio[0, 0].cpu().numpy()
    
    # Save
    output_path = f'examples/example_{speaker_name}.wav'
    sf.write(output_path, audio, config['data']['sampling_rate'])
    print(f"  ✅ Saved: {output_path} ({len(audio)/config['data']['sampling_rate']:.2f}s)")

print("\n✅ All examples generated!")
