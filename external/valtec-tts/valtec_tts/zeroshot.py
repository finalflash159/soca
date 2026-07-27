"""
Valtec TTS - Zero-Shot Vietnamese Voice Cloning API

Usage:
    from valtec_tts import ZeroShotTTS
    
    tts = ZeroShotTTS()
    audio, sr = tts.synthesize(
        text="Xin chào các bạn",
        reference_audio="path/to/reference.wav"
    )
    tts.clone_voice(
        text="Xin chào",
        reference_audio="path/to/reference.wav",
        output_path="output.wav"
    )
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


DEFAULT_ZEROSHOT_HF_REPO = "valtecAI-team/valtec-zeroshot-voice-cloning"
DEFAULT_ZEROSHOT_MODEL_NAME = "zeroshot-vietnamese"


def _get_cache_dir() -> Path:
    if os.name == 'nt':
        cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    cache_dir = cache_base / 'valtec_tts' / 'models'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class ZeroShotTTS:
    """
    Zero-shot Vietnamese voice cloning interface.

    Clone any voice from 3-10 seconds of reference audio.

    Example:
        tts = ZeroShotTTS()
        audio, sr = tts.synthesize("Xin chào", reference_audio="voice.wav")
        tts.clone_voice("Xin chào", "voice.wav", output_path="output.wav")
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize zero-shot voice cloning engine.

        Args:
            checkpoint_path: Path to checkpoint file. If None, auto-downloads.
            config_path: Path to config.json. If None, auto-resolved.
            device: 'cuda', 'cpu', or 'auto'.
        """
        import torch

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve model paths
        if checkpoint_path is None or config_path is None:
            model_dir = self._ensure_model_available()
            if checkpoint_path is None:
                ckpts = sorted(Path(model_dir).glob("G_*.pth"))
                if not ckpts:
                    raise FileNotFoundError(f"No checkpoint found in {model_dir}")
                checkpoint_path = str(ckpts[-1])
            if config_path is None:
                config_path = str(Path(model_dir) / "config.json")

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self._engine = None
        self._load_model()

    def _ensure_model_available(self) -> str:
        """Check local cache or download from HuggingFace."""
        # First check if pretrained/zeroshot exists locally (dev mode)
        package_root = Path(__file__).parent.parent
        local_dir = package_root / "pretrained" / "zeroshot"
        if (local_dir / "config.json").exists() and list(local_dir.glob("G_*.pth")):
            print(f"Using local model from: {local_dir}")
            return str(local_dir)

        # Check cache
        cache_dir = _get_cache_dir()
        model_dir = cache_dir / DEFAULT_ZEROSHOT_MODEL_NAME
        if (model_dir / "config.json").exists() and list(model_dir.glob("G_*.pth")):
            print(f"Using cached model from: {model_dir}")
            return str(model_dir)

        # Download
        print(f"Downloading zero-shot model from: {DEFAULT_ZEROSHOT_HF_REPO}")
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub required for auto-download. "
                "pip install huggingface_hub"
            )

        model_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["G_175000.pth", "config.json"]:
            hf_hub_download(
                repo_id=DEFAULT_ZEROSHOT_HF_REPO,
                filename=f"pretrained/zeroshot/{fname}",
                local_dir=str(model_dir),
                repo_type="space",
            )
        # Also need hasp weights
        hasp_dir = model_dir.parent / "hasp"
        hasp_dir.mkdir(parents=True, exist_ok=True)
        if not (hasp_dir / "pytorch_model.bin").exists():
            hf_hub_download(
                repo_id=DEFAULT_ZEROSHOT_HF_REPO,
                filename="pretrained/hasp/pytorch_model.bin",
                local_dir=str(hasp_dir),
                repo_type="space",
            )

        print("Download complete!")
        return str(model_dir)

    def _load_model(self):
        """Load all model components."""
        import json
        import torch
        from torch import nn

        package_root = Path(__file__).parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        from src.models import (
            SynthesizerZeroShot, SpeakerEncoder,
            StyleEncoder, ProsodyPredictor
        )
        from src.text.symbols import symbols

        with open(self.config_path, encoding='utf-8') as f:
            config = json.load(f)

        class HParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    if isinstance(v, dict) and k != 'spk2id':
                        v = HParams(**v)
                    setattr(self, k, v)

        self.hps = HParams(**config)
        self.sampling_rate = self.hps.data.sampling_rate

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        # Speaker Encoder
        self.speaker_encoder = SpeakerEncoder(
            device=self.device,
            embed_dim=getattr(self.hps.model, 'gin_channels', 512)
        )

        # Style Encoder
        self.style_encoder = StyleEncoder(
            n_mel_channels=80, style_dim=128
        ).to(self.device).eval()
        if 'prosody_encoder' in checkpoint:
            state = {k.replace('module.', ''): v
                     for k, v in checkpoint['prosody_encoder'].items()}
            self.style_encoder.load_state_dict(state)

        # Prosody Predictor
        self.prosody_predictor = ProsodyPredictor(
            style_dim=128, d_hid=256,
            text_dim=self.hps.model.hidden_channels, dropout=0.1
        ).to(self.device).eval()
        if 'prosody_predictor' in checkpoint:
            state = {k.replace('module.', ''): v
                     for k, v in checkpoint['prosody_predictor'].items()}
            self.prosody_predictor.load_state_dict(state)

        # Synthesizer
        self.model = SynthesizerZeroShot(
            n_vocab=len(symbols),
            spec_channels=self.hps.data.filter_length // 2 + 1,
            segment_size=config['train']['segment_size'] // self.hps.data.hop_length,
            inter_channels=self.hps.model.inter_channels,
            hidden_channels=self.hps.model.hidden_channels,
            filter_channels=self.hps.model.filter_channels,
            n_heads=self.hps.model.n_heads,
            n_layers=self.hps.model.n_layers,
            kernel_size=self.hps.model.kernel_size,
            p_dropout=self.hps.model.p_dropout,
            resblock=self.hps.model.resblock,
            resblock_kernel_sizes=self.hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hps.model.resblock_dilation_sizes,
            upsample_rates=self.hps.model.upsample_rates,
            upsample_initial_channel=self.hps.model.upsample_initial_channel,
            upsample_kernel_sizes=self.hps.model.upsample_kernel_sizes,
            gin_channels=getattr(self.hps.model, 'gin_channels', 512),
            prosody_dim=128,
            use_sdp=True,
            num_languages=config.get('num_languages', 8),
            num_tones=config.get('num_tones', 24),
            use_transformer_flow=getattr(
                self.hps.model, 'use_transformer_flow', True
            ),
        ).to(self.device).eval()

        state_dict = {k.replace('module.', ''): v
                      for k, v in checkpoint['model'].items()}
        missing, _ = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            for name, param in self.model.named_parameters():
                if name in missing and "films" in name:
                    nn.init.constant_(param, 0.0)

        del checkpoint
        print(f"[ZeroShotTTS] Ready on {self.device}")

    def extract_embeddings(self, audio_path: str):
        """Extract speaker and prosody embeddings from reference audio."""
        import torch
        import librosa
        from src.nn.mel_processing import mel_spectrogram_torch

        ref_audio, ref_sr = librosa.load(audio_path, sr=None)
        ref_t = torch.from_numpy(ref_audio).float().unsqueeze(0).to(self.device)

        speaker_emb = self.speaker_encoder(ref_t, sr=ref_sr)

        ref_24k = librosa.resample(
            ref_audio, orig_sr=ref_sr, target_sr=self.sampling_rate
        )
        ref_24k_t = torch.from_numpy(ref_24k).float().unsqueeze(0).to(self.device)
        mel = mel_spectrogram_torch(
            ref_24k_t, self.hps.data.filter_length, 80,
            self.sampling_rate, self.hps.data.hop_length,
            self.hps.data.win_length, 0, None
        )
        with torch.no_grad():
            prosody_emb = self.style_encoder(mel)

        return speaker_emb, prosody_emb

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech cloning a reference voice.

        Args:
            text: Vietnamese text to synthesize.
            reference_audio: Path to reference audio file (3-10 seconds).
            noise_scale: Controls voice variability.
            noise_scale_w: Controls duration variability.
            length_scale: Speed (1.0 = normal, < 1.0 = faster).

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        import torch
        from src.text import cleaned_text_to_sequence
        from src.vietnamese.text_processor import process_vietnamese_text
        from src.vietnamese.phonemizer import text_to_phonemes
        from src.nn import commons

        speaker_emb, prosody_emb = self.extract_embeddings(reference_audio)

        processed = process_vietnamese_text(text)
        phones, tones_raw, word2ph = text_to_phonemes(processed)
        phone_ids, tone_ids, language_ids = cleaned_text_to_sequence(
            phones, tones_raw, "VI"
        )

        if getattr(self.hps.data, 'add_blank', True):
            phone_ids = commons.intersperse(phone_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            language_ids = commons.intersperse(language_ids, 0)

        phone_t = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        tone_t = torch.LongTensor(tone_ids).unsqueeze(0).to(self.device)
        lang_t = torch.LongTensor(language_ids).unsqueeze(0).to(self.device)
        phone_len = torch.LongTensor([phone_t.shape[1]]).to(self.device)
        bert = torch.zeros(1, 1024, phone_t.shape[1]).to(self.device)
        ja_bert = torch.zeros(1, 768, phone_t.shape[1]).to(self.device)
        g = speaker_emb.unsqueeze(-1)

        with torch.no_grad():
            o, *_ = self.model.infer(
                phone_t, phone_len, sid=None,
                tone=tone_t, language=lang_t,
                bert=bert, ja_bert=ja_bert,
                g=g, prosody=prosody_emb,
                prosody_predictor=self.prosody_predictor,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )

        audio = o[0, 0].cpu().numpy()
        return audio, self.sampling_rate

    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        output_path: str = "output.wav",
        **kwargs
    ) -> str:
        """
        Clone voice and save to file.

        Args:
            text: Vietnamese text to synthesize.
            reference_audio: Path to reference audio (3-10 seconds).
            output_path: Path to save the output audio.
            **kwargs: Additional args passed to synthesize().

        Returns:
            Path to saved audio file.
        """
        audio, sr = self.synthesize(text, reference_audio, **kwargs)
        import soundfile as sf
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr)
        print(f"Saved: {output_path}")
        return output_path

    def __repr__(self) -> str:
        return f"ZeroShotTTS(device='{self.device}')"
