#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="valtec-tts",
    version="1.0.0",
    author="Valtec Team",
    author_email="contact@valtec.com",
    description="Vietnamese Text-to-Speech system with simple API and auto-download",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/valtec-tts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: Free For Home Use",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=2.0.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.0",
        "librosa>=0.9.0",
        "tqdm>=4.60.0",
        "Unidecode>=1.3.0",
        "num2words>=0.5.10",
        "inflect>=6.0.0",
        "cn2an>=0.5.20",
        "jieba>=0.42.0",
        "pypinyin>=0.44.0",
        "jamo>=0.4.1",
        "gruut>=2.4.0",
        "g2p-en>=2.1.0",
        "anyascii>=0.3.0",
        "viphoneme>=3.0.0",
        "underthesea>=8.0.0",
        "vinorm>=2.0.0",
        "huggingface_hub>=0.20.0",
        "eng-to-ipa>=0.0.2",
        "gradio>=5.0.0",
    ],
    extras_require={
        "play": ["sounddevice>=0.4.0"],
    },
    entry_points={
        "console_scripts": [
            "valtec-tts=infer:main",
            "valtec-tts-demo=demo_gradio:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt"],
    },
    zip_safe=False,
)
