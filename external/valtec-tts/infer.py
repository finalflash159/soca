#!/usr/bin/env python3


"""
Vietnamese TTS Inference Script
Synthesizes speech from text using trained model.
"""

import os
import sys
import json
import argparse
import glob
import re
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Local imports
from src.vietnamese.text_processor import process_vietnamese_text
from src.vietnamese.phonemizer import text_to_phonemes, VIPHONEME_AVAILABLE
from src.models.synthesizer import SynthesizerTrn
from src.text.symbols import symbols
from src.utils import helpers as utils

def find_latest_checkpoint(model_dir, prefix="G"):
    """Find the latest checkpoint in model directory."""
    pattern = os.path.join(model_dir, f"{prefix}*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    def get_step(path):
        match = re.search(rf'{prefix}(\d+)\.pth', path)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Vietnamese TTS Inference")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to generator checkpoint (G*.pth). If not specified, uses latest from --model_dir")
    parser.add_argument("--model_dir", type=str, default="./pretrained",
                        help="Model directory to find latest checkpoint (default: ./logs/vietnamese_10ch_finetune_viphoneme)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (auto-detect if not specified)")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text to synthesize")
    parser.add_argument("--speaker", "-s", type=str, default=None,
                        help="Speaker name (from config)")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file with texts (one per line)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for batch mode")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--sdp_ratio", type=float, default=0.0,
                        help="SDP ratio (0.0 = deterministic, 1.0 = stochastic)")
    parser.add_argument("--noise_scale", type=float, default=0.667,
                        help="Noise scale for generation")
    parser.add_argument("--noise_scale_w", type=float, default=0.8,
                        help="Noise scale for duration")
    parser.add_argument("--length_scale", type=float, default=1.0,
                        help="Length scale (speed)")
    return parser.parse_args()


class VietnameseTTS:
    """Vietnamese TTS synthesizer using trained VITS-based model."""
    
    def __init__(self, checkpoint_path, config_path, device="cuda"):
        self.device = device
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.sampling_rate = self.config['data']['sampling_rate']
        self.spk2id = self.config['data']['spk2id']
        self.speakers = list(self.spk2id.keys())
        self.add_blank = self.config['data'].get('add_blank', True)
        
        print(f"Available speakers: {self.speakers}")
        
        # Load model
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path):
        """Load the trained model."""
        
        # Create model
        hps_data = utils.HParams(**self.config['data'])
        hps_model = utils.HParams(**self.config['model'])
        
        self.model = SynthesizerTrn(
            len(symbols),
            self.config['data']['filter_length'] // 2 + 1,
            self.config['train']['segment_size'] // self.config['data']['hop_length'],
            n_speakers=self.config['data']['n_speakers'],
            **self.config['model'],
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Handle DDP checkpoint
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
    
    def text_to_sequence(self, text, speaker):
        """Convert text to model input tensors."""
        from src.text import cleaned_text_to_sequence
        from src.nn import commons
        
        # Normalize text
        normalized_text = process_vietnamese_text(text)
        
        # Convert to phonemes
        phones, tones, word2ph = text_to_phonemes(normalized_text, use_viphoneme=VIPHONEME_AVAILABLE)
        
        # Convert to sequence
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, "VI")
        
        # Add blanks if needed
        if self.add_blank:
            phone_ids = commons.intersperse(phone_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            lang_ids = commons.intersperse(lang_ids, 0)
        
        # Get speaker ID
        if speaker not in self.spk2id:
            print(f"Warning: Speaker '{speaker}' not found, using first speaker: {self.speakers[0]}")
            speaker = self.speakers[0]
        speaker_id = self.spk2id[speaker]
        
        # Create tensors
        x = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(phone_ids)]).to(self.device)
        tone = torch.LongTensor(tone_ids).unsqueeze(0).to(self.device)
        language = torch.LongTensor(lang_ids).unsqueeze(0).to(self.device)
        sid = torch.LongTensor([speaker_id]).to(self.device)
        
        # Create dummy BERT features (zeros if disabled)
        bert = torch.zeros(1024, len(phone_ids)).unsqueeze(0).to(self.device)
        ja_bert = torch.zeros(768, len(phone_ids)).unsqueeze(0).to(self.device)
        
        return x, x_lengths, tone, language, sid, bert, ja_bert
    
    @torch.no_grad()
    def synthesize(self, text, speaker, sdp_ratio=0.0, noise_scale=0.667, 
                   noise_scale_w=0.8, length_scale=1.0):
        """
        Synthesize speech from text.
        
        Args:
            text: Input Vietnamese text
            speaker: Speaker name
            sdp_ratio: Stochastic duration predictor ratio (0=deterministic)
            noise_scale: Noise scale for generation
            noise_scale_w: Noise scale for duration
            length_scale: Speed control (1.0=normal, <1.0=faster, >1.0=slower)
        
        Returns:
            audio: numpy array of audio samples
            sr: sample rate
        """
        # Prepare inputs
        x, x_lengths, tone, language, sid, bert, ja_bert = self.text_to_sequence(text, speaker)
        
        # Generate
        audio, attn, *_ = self.model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        
        audio = audio[0, 0].cpu().numpy()
        
        return audio, self.sampling_rate
    
    def save_audio(self, audio, sr, output_path):
        """Save audio to file."""
        sf.write(output_path, audio, sr)
        print(f"Audio saved to {output_path}")


def _extract_iter_from_checkpoint(checkpoint_path: str) -> str | None:
    base = os.path.basename(checkpoint_path)
    m = re.search(r"G_(\d+)\.pth$", base)
    if m:
        return m.group(1)
    return None


def _append_suffix_before_ext(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _resolve_output_path(output: str, output_dir: str, suffix: str) -> Path:
    p = Path(output)
    if not p.is_absolute():
        p = Path(output_dir) / p.name
    p = _append_suffix_before_ext(p, suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main():
    args = parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Find checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.model_dir, "G")
        if checkpoint_path is None:
            # Try to download from Hugging Face
            print(f"No checkpoint found in {args.model_dir}")
            print("Attempting to download from Hugging Face...")
            try:
                from huggingface_hub import snapshot_download
                
                # Default HF repo
                hf_repo = "valtecAI-team/valtec-tts-pretrained"
                
                # Get cache directory
                if os.name == 'nt':  # Windows
                    cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
                else:  # Linux/Mac
                    cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
                
                model_dir = cache_base / 'valtec_tts' / 'models' / 'vits-vietnamese'
                model_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Downloading model to: {model_dir}")
                snapshot_download(repo_id=hf_repo, local_dir=str(model_dir))
                print("Download complete!")
                
                # Update model_dir and find checkpoint
                args.model_dir = str(model_dir)
                checkpoint_path = find_latest_checkpoint(args.model_dir, "G")
                
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please specify --checkpoint or --model_dir")
                return
            
            if checkpoint_path is None:
                print("Error: Could not find checkpoint after download")
                return
                
        print(f"Using latest checkpoint: {checkpoint_path}")

    iter_str = _extract_iter_from_checkpoint(checkpoint_path)
    iter_suffix = f"iter{iter_str}" if iter_str is not None else "iterunknown"
    
    # Auto-find config if in same directory as checkpoint
    config_path = args.config
    if config_path is None:
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Error: config.json not found at {config_path}")
            return
        print(f"Using config: {config_path}")
    
    # Initialize TTS
    print("Loading model...")
    tts = VietnameseTTS(checkpoint_path, config_path, args.device)
    
    # Get default speaker
    default_speaker = args.speaker or tts.speakers[0]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Vietnamese TTS - Interactive Mode")
        print(f"Default speaker: {default_speaker}")
        print(f"Available speakers: {', '.join(tts.speakers)}")
        print("Commands: 'quit' to exit, 'speaker NAME' to change speaker")
        print("=" * 60 + "\n")
        
        current_speaker = default_speaker
        
        while True:
            try:
                text = input("Enter text: ").strip()
                
                if not text:
                    continue
                
                if text.lower() == 'quit':
                    break
                
                if text.lower().startswith('speaker '):
                    new_speaker = text[8:].strip()
                    if new_speaker in tts.speakers:
                        current_speaker = new_speaker
                        print(f"Speaker changed to: {current_speaker}")
                    else:
                        print(f"Speaker not found. Available: {', '.join(tts.speakers)}")
                    continue
                
                # Synthesize
                print(f"Synthesizing with speaker '{current_speaker}'...")
                audio, sr = tts.synthesize(
                    text, current_speaker,
                    sdp_ratio=args.sdp_ratio,
                    noise_scale=args.noise_scale,
                    noise_scale_w=args.noise_scale_w,
                    length_scale=args.length_scale,
                )
                
                # Save with timestamp
                import time
                output_path = _resolve_output_path(
                    f"output_{int(time.time())}.wav",
                    str(output_dir),
                    iter_suffix,
                )
                tts.save_audio(audio, sr, str(output_path))
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.input_file:
        # Batch mode
        print(f"\nBatch processing from {args.input_file}")
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        for i, text in enumerate(tqdm(lines, desc="Synthesizing")):
            try:
                # Check for speaker specification: "speaker|text"
                if '|' in text:
                    speaker, text = text.split('|', 1)
                    speaker = speaker.strip()
                else:
                    speaker = default_speaker
                
                audio, sr = tts.synthesize(
                    text, speaker,
                    sdp_ratio=args.sdp_ratio,
                    noise_scale=args.noise_scale,
                    noise_scale_w=args.noise_scale_w,
                    length_scale=args.length_scale,
                )
                
                output_path = _resolve_output_path(
                    f"{i:04d}.wav",
                    str(output_dir),
                    iter_suffix,
                )
                tts.save_audio(audio, sr, str(output_path))
                
            except Exception as e:
                print(f"Error processing line {i}: {e}")
        
        print(f"\nBatch processing complete. Outputs saved to {output_dir}")
    
    elif args.text:
        # Single text mode
        print(f"\nSynthesizing: {args.text}")
        print(f"Speaker: {default_speaker}")
        
        audio, sr = tts.synthesize(
            args.text, default_speaker,
            sdp_ratio=args.sdp_ratio,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )
        
        output_path = _resolve_output_path(args.output, str(output_dir), iter_suffix)
        tts.save_audio(audio, sr, str(output_path))
    
    else:
        print("Please provide --text, --input_file, or --interactive")
        print("Example: python infer.py --checkpoint G_10000.pth --config config.json --text 'Xin chào'")


if __name__ == "__main__":
    main()
