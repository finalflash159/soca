#!/usr/bin/env python3
"""
VALTEC TTS Benchmark Script
Measures inference time on CPU and CUDA, counts model parameters.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np

# Local imports
from src.vietnamese.text_processor import process_vietnamese_text
from src.vietnamese.phonemizer import text_to_phonemes, VIPHONEME_AVAILABLE
from src.models.synthesizer import SynthesizerTrn
from src.text.symbols import symbols
from src.utils import helpers as utils
from infer import find_latest_checkpoint


def count_parameters(model):
    """Count total and trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_params(num):
    """Format parameter count (e.g., 35.5M)."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def load_model(checkpoint_path, config_path, device):
    """Load model to specified device."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model = SynthesizerTrn(
        len(symbols),
        config['data']['filter_length'] // 2 + 1,
        config['train']['segment_size'] // config['data']['hop_length'],
        n_speakers=config['data']['n_speakers'],
        **config['model'],
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, config


def prepare_input(text, config, device):
    """Prepare input tensors for inference."""
    from src.text import cleaned_text_to_sequence
    from src.nn import commons
    
    add_blank = config['data'].get('add_blank', True)
    spk2id = config['data']['spk2id']
    speakers = list(spk2id.keys())
    speaker = speakers[0]
    
    # Normalize text
    normalized_text = process_vietnamese_text(text)
    
    # Convert to phonemes
    phones, tones, word2ph = text_to_phonemes(normalized_text, use_viphoneme=VIPHONEME_AVAILABLE)
    
    # Convert to sequence
    phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, "VI")
    
    # Add blanks if needed
    if add_blank:
        phone_ids = commons.intersperse(phone_ids, 0)
        tone_ids = commons.intersperse(tone_ids, 0)
        lang_ids = commons.intersperse(lang_ids, 0)
    
    speaker_id = spk2id[speaker]
    
    # Create tensors
    x = torch.LongTensor(phone_ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(phone_ids)]).to(device)
    tone = torch.LongTensor(tone_ids).unsqueeze(0).to(device)
    language = torch.LongTensor(lang_ids).unsqueeze(0).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)
    bert = torch.zeros(1024, len(phone_ids)).unsqueeze(0).to(device)
    ja_bert = torch.zeros(768, len(phone_ids)).unsqueeze(0).to(device)
    
    return x, x_lengths, tone, language, sid, bert, ja_bert


@torch.no_grad()
def run_inference(model, inputs, n_runs=10, warmup=3):
    """Run inference multiple times and measure time."""
    x, x_lengths, tone, language, sid, bert, ja_bert = inputs
    
    # Warmup
    for _ in range(warmup):
        _ = model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            sdp_ratio=0.0, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0
        )
    
    # Sync CUDA if needed
    if x.is_cuda:
        torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        audio, *_ = model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            sdp_ratio=0.0, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0
        )
        if x.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    audio_length = audio.shape[-1] / 44100  # Assume 44.1kHz
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'audio_length': audio_length,
        'rtf': np.mean(times) / audio_length  # Real-time factor
    }


def get_gpu_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_compute = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
        return {
            'name': gpu_name,
            'memory_gb': gpu_memory,
            'compute_capability': f"{gpu_compute[0]}.{gpu_compute[1]}"
        }
    return None


def get_cpu_info():
    """Get CPU information."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    return line.split(':')[1].strip()
    except:
        pass
    return "Unknown CPU"


def ensure_model_available(model_dir):
    """Check if model exists locally, otherwise download from Hugging Face."""
    from pathlib import Path
    
    checkpoint_path = find_latest_checkpoint(model_dir, "G")
    if checkpoint_path is not None:
        return checkpoint_path, os.path.join(os.path.dirname(checkpoint_path), "config.json")
    
    # Try to download from Hugging Face
    print(f"No checkpoint found in {model_dir}")
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
        
        model_dir_path = cache_base / 'valtec_tts' / 'models' / 'vits-vietnamese'
        model_dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading model to: {model_dir_path}")
        snapshot_download(repo_id=hf_repo, local_dir=str(model_dir_path))
        print("Download complete!")
        
        # Find checkpoint after download
        checkpoint_path = find_latest_checkpoint(str(model_dir_path), "G")
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint found after download")
        
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        return checkpoint_path, config_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


def main():
    parser = argparse.ArgumentParser(description="VALTEC TTS Benchmark")
    parser.add_argument("--model_dir", type=str, default="./pretrained",
                        help="Model directory")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of inference runs for timing")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    args = parser.parse_args()
    
    # Test texts
    test_texts = [
        "Xin chào các bạn",  # Short
        "Hôm nay thời tiết rất đẹp, trời trong xanh và nắng ấm áp.",  # Medium
        "Việt Nam là một quốc gia nằm ở vùng Đông Nam Á, có lịch sử và văn hóa lâu đời với nhiều di sản được UNESCO công nhận.",  # Long
    ]
    
    text_labels = ["Short (15 chars)", "Medium (52 chars)", "Long (120 chars)"]
    
    # Find or download checkpoint
    try:
        checkpoint_path, config_path = ensure_model_available(args.model_dir)
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using config: {config_path}")
    
    print("=" * 70)
    print("VALTEC TTS Benchmark")
    print("=" * 70)
    
    # System Info
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    
    print(f"\n📌 System Information:")
    print(f"   CPU: {cpu_info}")
    if gpu_info:
        print(f"   GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
        print(f"   CUDA Compute Capability: {gpu_info['compute_capability']}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    # Model parameters (load once to count)
    print(f"\n🔧 Loading model for parameter count...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(checkpoint_path, config_path, device)
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {format_params(total_params)} ({total_params:,})")
    print(f"   Trainable: {format_params(trainable_params)} ({trainable_params:,})")
    print(f"   Sample Rate: {config['data']['sampling_rate']} Hz")
    
    results = {'cpu': [], 'cuda': []}
    
    # Benchmark CPU
    print(f"\n🖥️  CPU Benchmark (Intel i5-14500):")
    print("-" * 50)
    
    model_cpu, _ = load_model(checkpoint_path, config_path, "cpu")
    
    for text, label in zip(test_texts, text_labels):
        inputs = prepare_input(text, config, "cpu")
        result = run_inference(model_cpu, inputs, args.n_runs, args.warmup)
        results['cpu'].append(result)
        print(f"   {label}:")
        print(f"      Inference: {result['mean']*1000:.2f}ms ± {result['std']*1000:.2f}ms")
        print(f"      Audio length: {result['audio_length']:.2f}s")
        print(f"      RTF: {result['rtf']:.4f} {'✅ Real-time' if result['rtf'] < 1 else '❌ Slower than real-time'}")
    
    del model_cpu
    
    # Benchmark CUDA if available
    if torch.cuda.is_available():
        print(f"\n🚀 CUDA Benchmark ({gpu_info['name']}):")
        print("-" * 50)
        
        model_cuda, _ = load_model(checkpoint_path, config_path, "cuda")
        
        for text, label in zip(test_texts, text_labels):
            inputs = prepare_input(text, config, "cuda")
            result = run_inference(model_cuda, inputs, args.n_runs, args.warmup)
            results['cuda'].append(result)
            print(f"   {label}:")
            print(f"      Inference: {result['mean']*1000:.2f}ms ± {result['std']*1000:.2f}ms")
            print(f"      Audio length: {result['audio_length']:.2f}s")
            print(f"      RTF: {result['rtf']:.4f} ✅ Real-time" if result['rtf'] < 1 else f"      RTF: {result['rtf']:.4f} ❌ Slower than real-time")
        
        # Speedup comparison
        print(f"\n⚡ GPU Speedup over CPU:")
        for i, label in enumerate(text_labels):
            speedup = results['cpu'][i]['mean'] / results['cuda'][i]['mean']
            print(f"   {label}: {speedup:.1f}x faster on GPU")
        
        del model_cuda
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
    
    # Return results for programmatic use
    return {
        'cpu_info': cpu_info,
        'gpu_info': gpu_info,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'results': results
    }


if __name__ == "__main__":
    main()
