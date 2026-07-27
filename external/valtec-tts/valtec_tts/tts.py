"""
Valtec TTS - Simple Vietnamese Text-to-Speech API

Usage:
    from valtec_tts import TTS
    
    tts = TTS()
    tts.speak("Xin chào các bạn", output_path="output.wav")
    
    # Or get audio directly
    audio, sr = tts.synthesize("Xin chào")
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import json

import numpy as np

# Hugging Face Hub for model download
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


# Default model repository on Hugging Face
DEFAULT_HF_REPO = "valtecAI-team/valtec-tts-pretrained"
DEFAULT_MODEL_NAME = "vits-vietnamese"

# Local cache directory
def get_cache_dir() -> Path:
    """Get the cache directory for storing models."""
    # Use standard cache locations
    if os.name == 'nt':  # Windows
        cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:  # Linux/Mac
        cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    
    cache_dir = cache_base / 'valtec_tts' / 'models'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class TTS:
    """
    Simple Vietnamese Text-to-Speech interface.
    
    Example:
        tts = TTS()
        tts.speak("Xin chào", output_path="hello.wav")
        
        # Or get audio array
        audio, sr = tts.synthesize("Xin chào")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        hf_repo: str = DEFAULT_HF_REPO,
    ):
        """
        Initialize TTS engine.
        
        Args:
            model_path: Path to local model directory. If None, auto-downloads from Hugging Face.
            device: Device to use ('cuda', 'cpu', or 'auto' for automatic detection).
            hf_repo: Hugging Face repository ID for model download.
        """
        self.hf_repo = hf_repo
        
        # Determine device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Get model path
        if model_path is None:
            model_path = self._ensure_model_available()
        
        self.model_path = Path(model_path)
        self._engine = None
        self._load_model()
    
    def _ensure_model_available(self) -> str:
        """Ensure model is available locally, download if not."""
        cache_dir = get_cache_dir()
        model_dir = cache_dir / DEFAULT_MODEL_NAME
        config_path = model_dir / "config.json"
        
        # Check if model already exists
        if config_path.exists():
            # Find checkpoint
            checkpoints = list(model_dir.glob("G_*.pth"))
            if checkpoints:
                print(f"Using cached model from: {model_dir}")
                return str(model_dir)
        
        # Need to download
        print(f"Model not found locally. Downloading from Hugging Face: {self.hf_repo}")
        return self._download_model(model_dir)
    
    def _download_model(self, target_dir: Path) -> str:
        """Download model from Hugging Face Hub."""
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is required for auto-download. "
                "Install with: pip install huggingface_hub"
            )
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download entire model directory
            print(f"Downloading model to: {target_dir}")
            snapshot_download(
                repo_id=self.hf_repo,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            print("Download complete!")
            return str(target_dir)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from {self.hf_repo}: {e}\n"
                "Please check your internet connection or provide a local model_path."
            )
    
    def _load_model(self):
        """Load the TTS model."""
        # Add parent directory to path for imports
        package_root = Path(__file__).parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        try:
            from infer import VietnameseTTS, find_latest_checkpoint
        except ImportError as e:
            raise ImportError(
                f"Failed to import infer module: {e}\n"
                "This usually means you're running the package from outside the project directory.\n"
                "Please run from the project root or ensure the project is properly set up."
            ) from e

        # Find checkpoint and config
        checkpoint = find_latest_checkpoint(str(self.model_path), "G")
        config_path = self.model_path / "config.json"

        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {self.model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")

        print(f"Loading model from: {checkpoint}")
        self._engine = VietnameseTTS(checkpoint, str(config_path), self.device)

        # Store speakers
        self.speakers = self._engine.speakers
        self.default_speaker = self.speakers[0] if self.speakers else None
        print(f"Available speakers: {self.speakers}")
    
    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Vietnamese text to synthesize.
            speaker: Speaker name. Uses default if not specified.
            speed: Speech speed (1.0 = normal, < 1.0 = faster, > 1.0 = slower).
            noise_scale: Controls voice variability.
            noise_scale_w: Controls duration variability.
            sdp_ratio: Stochastic duration predictor ratio (0 = deterministic).
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._engine is None:
            raise RuntimeError("Model not loaded")
        
        speaker = speaker or self.default_speaker
        
        audio, sr = self._engine.synthesize(
            text=text,
            speaker=speaker,
            length_scale=speed,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            sdp_ratio=sdp_ratio,
        )
        
        return audio, sr
    
    def speak(
        self,
        text: str,
        output_path: str = "output.wav",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        play: bool = False,
        **kwargs
    ) -> str:
        """
        Synthesize and save speech to file.
        
        Args:
            text: Vietnamese text to synthesize.
            output_path: Path to save the audio file.
            speaker: Speaker name. Uses default if not specified.
            speed: Speech speed (1.0 = normal).
            play: If True, attempt to play the audio (requires sounddevice).
            **kwargs: Additional arguments passed to synthesize().
        
        Returns:
            Path to the saved audio file.
        """
        audio, sr = self.synthesize(text, speaker=speaker, speed=speed, **kwargs)
        
        # Save audio
        import soundfile as sf
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr)
        print(f"Audio saved to: {output_path}")
        
        # Optionally play audio
        if play:
            try:
                import sounddevice as sd
                sd.play(audio, sr)
                sd.wait()
            except ImportError:
                print("Install sounddevice to play audio: pip install sounddevice")
        
        return str(output_path)
    
    def list_speakers(self) -> list:
        """Get list of available speakers."""
        return self.speakers
    
    def __repr__(self) -> str:
        return f"TTS(device='{self.device}', speakers={self.speakers})"
