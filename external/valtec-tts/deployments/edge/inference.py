#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese TTS Edge Inference using ONNX Runtime
Optimized for edge devices and lightweight deployment.

Uses viphoneme library for accurate Vietnamese G2P conversion.
Auto-downloads ONNX models from HuggingFace Hub on first run.
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import onnxruntime as ort

# Suppress warnings
warnings.filterwarnings("ignore")

# Check viphoneme availability
try:
    from viphoneme import vi2IPA
    VIPHONEME_AVAILABLE = True
except ImportError:
    VIPHONEME_AVAILABLE = False
    print("⚠️ viphoneme not installed. Run: pip install viphoneme")


def download_onnx_models(model_dir: Optional[str] = None) -> str:
    """
    Download ONNX models from HuggingFace Hub if not already cached.
    
    Args:
        model_dir: Optional custom directory for models
        
    Returns:
        Path to model directory
    """
    if model_dir and Path(model_dir).exists():
        # Check if all required files exist
        required_files = ['text_encoder.onnx', 'duration_predictor.onnx', 
                         'flow.onnx', 'decoder.onnx', 'tts_config.json']
        if all((Path(model_dir) / f).exists() for f in required_files):
            print(f"Using local models from: {model_dir}")
            return str(model_dir)
    
    # Download from HuggingFace Hub
    try:
        from huggingface_hub import snapshot_download
        
        hf_repo = "valtecAI-team/valtec-tts-onnx"
        
        # Determine cache directory
        if os.name == 'nt':  # Windows
            cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        else:  # Linux/Mac
            cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        
        cache_dir = cache_base / 'valtec_tts' / 'onnx_models'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading ONNX models from {hf_repo}...")
        model_path = snapshot_download(
            repo_id=hf_repo,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False
        )
        print(f"✅ Models cached to: {model_path}")
        return model_path
        
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        print("   Or manually download models to a directory and specify --model-dir")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        if model_dir:
            print(f"   Trying to use specified directory: {model_dir}")
            return str(model_dir)
        sys.exit(1)


class VietnameseG2P:
    """Vietnamese Grapheme-to-Phoneme converter using viphoneme library."""
    
    # viphoneme tone mapping: 1=ngang, 2=huyền, 3=ngã, 4=hỏi, 5=sắc, 6=nặng
    # Internal tone: 0=ngang, 1=sắc, 2=huyền, 3=ngã, 4=hỏi, 5=nặng
    VIPHONEME_TONE_MAP = {1: 0, 2: 2, 3: 3, 4: 4, 5: 1, 6: 5}
    VI_TONE_OFFSET = 16  # Vietnamese tone start offset
    
    # Modifier characters to skip
    SKIP_CHARS = {'\u0306', '\u0361', '\u032f', '\u0330', '\u0329', 'ʷ', 'ʰ', 'ː'}
    
    def __init__(self, symbol_to_id: dict, vi_lang_id: int = 7):
        self.symbol_to_id = symbol_to_id
        self.vi_lang_id = vi_lang_id
        
        if not VIPHONEME_AVAILABLE:
            raise ImportError("viphoneme library is required. Install with: pip install viphoneme")
    
    def text_to_phonemes(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Convert Vietnamese text to phoneme IDs using viphoneme library.
        
        Args:
            text: Vietnamese text to convert
            
        Returns:
            phonemes: List of phoneme IDs
            tones: List of tone IDs (with offset)
            languages: List of language IDs
        """
        phonemes = []
        tones = []
        languages = []
        
        # Get IPA from viphoneme
        ipa_text = vi2IPA(text).strip()
        
        if not ipa_text:
            # Empty result
            boundary_id = self.symbol_to_id.get('_', 0)
            return [boundary_id, boundary_id], [16, 16], [self.vi_lang_id, self.vi_lang_id]
        
        # Parse viphoneme output
        # Format: syllables separated by space, compound words joined by underscore
        # Tone number (1-6) at end of each syllable
        tokens = ipa_text.split()
        
        for token in tokens:
            # Handle punctuation-only tokens
            if all(c in ',.!?;:\'"()[]{}' or c == '.' for c in token):
                for c in token:
                    if c in self.symbol_to_id:
                        phonemes.append(self.symbol_to_id[c])
                        tones.append(0)
                        languages.append(self.vi_lang_id)
                continue
            
            # Split compound words by underscore
            syllables = token.split('_')
            
            for syllable in syllables:
                if not syllable:
                    continue
                
                syllable_phones = []
                syllable_tone = 0
                i = 0
                
                while i < len(syllable):
                    char = syllable[i]
                    
                    # Tone number at end
                    if char.isdigit():
                        syllable_tone = self.VIPHONEME_TONE_MAP.get(int(char), 0)
                        i += 1
                        continue
                    
                    # Skip combining marks
                    if ord(char) >= 0x0300 and ord(char) <= 0x036F:
                        i += 1
                        continue
                    
                    # Skip modifier letters
                    if char in self.SKIP_CHARS:
                        if syllable_phones:
                            syllable_phones[-1] = syllable_phones[-1] + char
                        i += 1
                        continue
                    
                    # Skip tie bars
                    if char in {'\u0361', '\u035c'}:
                        i += 1
                        continue
                    
                    # Regular phoneme character
                    syllable_phones.append(char)
                    i += 1
                
                # Map phonemes to IDs
                for ph in syllable_phones:
                    ph_id = self.symbol_to_id.get(ph, self.symbol_to_id.get('UNK', 305))
                    phonemes.append(ph_id)
                    tones.append(syllable_tone)
                    languages.append(self.vi_lang_id)
        
        # Add boundary tokens
        boundary_id = self.symbol_to_id.get('_', 0)
        phonemes = [boundary_id] + phonemes + [boundary_id]
        tones = [0] + tones + [0]
        languages = [self.vi_lang_id] + languages + [self.vi_lang_id]
        
        # Add tone offset
        tones = [t + self.VI_TONE_OFFSET for t in tones]
        
        return phonemes, tones, languages
    
    def add_blanks(self, phonemes: List[int], tones: List[int], languages: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """Add blank tokens between phonemes (required for TTS model)."""
        with_blanks = []
        tones_blanks = []
        langs_blanks = []
        
        for p, t, l in zip(phonemes, tones, languages):
            with_blanks.append(0)  # blank
            tones_blanks.append(0)
            langs_blanks.append(self.vi_lang_id)
            
            with_blanks.append(p)
            tones_blanks.append(t)
            langs_blanks.append(l)
        
        with_blanks.append(0)
        tones_blanks.append(0)
        langs_blanks.append(self.vi_lang_id)
        
        return with_blanks, tones_blanks, langs_blanks


class VietnameTTSEdge:
    """Vietnamese TTS using ONNX Runtime for edge deployment."""
    
    def __init__(self, model_dir: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize TTS engine.
        
        Args:
            model_dir: Optional path to directory containing ONNX models.
                      If not specified, models will be auto-downloaded from HuggingFace Hub.
            device: 'cpu' or 'cuda'
        """
        # Download/locate models
        self.model_dir = Path(download_onnx_models(model_dir))
        self.device = device
        
        # Load config
        config_path = self.model_dir / 'tts_config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config['sample_rate']
        self.symbol_to_id = self.config['symbol_to_id']
        self.vi_lang_id = self.config['language_id_map']['VI']
        
        # Initialize G2P with viphoneme
        self.g2p = VietnameseG2P(self.symbol_to_id, self.vi_lang_id)
        
        # Load ONNX models
        self._load_models()
        
        print(f"✅ TTS Engine initialized (device: {device}, viphoneme: {VIPHONEME_AVAILABLE})")
    
    def _load_models(self):
        """Load ONNX models."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        
        print("Loading ONNX models...")
        
        self.text_encoder = ort.InferenceSession(
            str(self.model_dir / 'text_encoder.onnx'),
            providers=providers
        )
        print("  ✓ text_encoder.onnx")
        
        self.duration_predictor = ort.InferenceSession(
            str(self.model_dir / 'duration_predictor.onnx'),
            providers=providers
        )
        print("  ✓ duration_predictor.onnx")
        
        self.flow = ort.InferenceSession(
            str(self.model_dir / 'flow.onnx'),
            providers=providers
        )
        print("  ✓ flow.onnx")
        
        self.decoder = ort.InferenceSession(
            str(self.model_dir / 'decoder.onnx'),
            providers=providers
        )
        print("  ✓ decoder.onnx")
    
    def synthesize(
        self,
        text: str,
        speaker_id: int = 1,
        noise_scale: float = 0.667,
        length_scale: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Vietnamese text to synthesize
            speaker_id: Speaker ID (0=NF, 1=SF, 2=NM1, 3=SM, 4=NM2)
                NF=Northern Female, SF=Southern Female, NM1/NM2=Northern Male, SM=Southern Male
            noise_scale: Controls voice variation (default 0.667)
            length_scale: Controls speech speed (>1 slower, <1 faster)
        
        Returns:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate (24000)
        """
        # Text to phonemes using viphoneme
        phonemes, tones, languages = self.g2p.text_to_phonemes(text)
        phonemes, tones, languages = self.g2p.add_blanks(phonemes, tones, languages)
        
        seq_len = len(phonemes)
        
        # Prepare inputs
        phone_ids = np.array([phonemes], dtype=np.int64)
        phone_lengths = np.array([seq_len], dtype=np.int64)
        tone_ids = np.array([tones], dtype=np.int64)
        language_ids = np.array([languages], dtype=np.int64)
        bert = np.zeros((1, 1024, seq_len), dtype=np.float32)
        ja_bert = np.zeros((1, 768, seq_len), dtype=np.float32)
        sid = np.array([speaker_id], dtype=np.int64)
        
        # Text encoder
        enc_outputs = self.text_encoder.run(None, {
            'phone_ids': phone_ids,
            'phone_lengths': phone_lengths,
            'tone_ids': tone_ids,
            'language_ids': language_ids,
            'bert': bert,
            'ja_bert': ja_bert,
            'speaker_id': sid
        })
        
        x_encoded, m_p, logs_p, x_mask, g = enc_outputs
        
        # Duration prediction
        dp_outputs = self.duration_predictor.run(None, {
            'x': x_encoded,
            'x_mask': x_mask,
            'g': g
        })
        logw = dp_outputs[0]
        
        # Compute durations
        durations = np.ceil(np.exp(logw) * x_mask * length_scale).astype(np.int32)
        total_frames = int(durations.sum())
        
        # Expand m_p and logs_p
        channels = m_p.shape[1]
        expanded_mp = np.zeros((1, channels, total_frames), dtype=np.float32)
        expanded_logs_p = np.zeros((1, channels, total_frames), dtype=np.float32)
        
        frame_idx = 0
        for t in range(durations.shape[2]):
            dur = int(durations[0, 0, t])
            for d in range(dur):
                if frame_idx < total_frames:
                    expanded_mp[0, :, frame_idx] = m_p[0, :, t]
                    expanded_logs_p[0, :, frame_idx] = logs_p[0, :, t]
                    frame_idx += 1
        
        # Sample z_p
        noise = np.random.randn(1, channels, total_frames).astype(np.float32) * noise_scale
        z_p = expanded_mp + np.exp(expanded_logs_p) * noise
        
        # Flow reverse
        y_mask = np.ones((1, 1, total_frames), dtype=np.float32)
        flow_outputs = self.flow.run(None, {
            'z_p': z_p,
            'y_mask': y_mask,
            'g': g
        })
        z = flow_outputs[0]
        
        # Decode
        dec_outputs = self.decoder.run(None, {
            'z': z,
            'g': g
        })
        audio = dec_outputs[0].squeeze()
        
        return audio, self.sample_rate


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vietnamese TTS Edge Inference (viphoneme + HF Hub)')
    parser.add_argument('--text', type=str, default='Xin chào, tôi là hệ thống tổng hợp giọng nói tiếng Việt.',
                        help='Text to synthesize')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Path to ONNX models (optional, auto-downloads from HF Hub if not specified)')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output audio file path')
    parser.add_argument('--speaker', type=int, default=1, choices=[0, 1, 2, 3, 4],
                        help='Speaker: 0=NF(Bắc-Nữ), 1=SF(Nam-Nữ), 2=NM1(Bắc-Nam1), 3=SM(Nam-Nam), 4=NM2(Bắc-Nam2)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speech speed (>1 slower, <1 faster)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Check viphoneme
    if not VIPHONEME_AVAILABLE:
        print("❌ Error: viphoneme library is required")
        print("   Install with: pip install viphoneme")
        sys.exit(1)
    
    # Initialize TTS
    tts = VietnameTTSEdge(args.model_dir, device=args.device)
    
    # Synthesize
    print(f"\nSynthesizing: \"{args.text}\"")
    audio, sr = tts.synthesize(
        text=args.text,
        speaker_id=args.speaker,
        length_scale=args.speed
    )
    
    # Save audio
    try:
        import soundfile as sf
        sf.write(args.output, audio, sr)
        print(f"✅ Saved to {args.output}")
    except ImportError:
        # Fallback to scipy
        from scipy.io import wavfile
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(args.output, sr, audio_int)
        print(f"✅ Saved to {args.output}")
    
    print(f"   Duration: {len(audio)/sr:.2f}s")


if __name__ == '__main__':
    main()
