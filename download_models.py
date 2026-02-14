#!/usr/bin/env python3
"""
Model Download Script for sherpa-onnx Voice Assistant

This script downloads all required models:
1. Silero VAD model
2. Whisper tiny multilingual model  
3. Piper TTS Hindi Rohan voice

Usage:
    python download_models.py [--models-dir ./models]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_file(url: str, output_path: str):
    """Download a file using wget or curl."""
    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")
    
    # Try wget first, then curl
    try:
        subprocess.run(["wget", "-q", "--show-progress", "-O", output_path, url], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(["curl", "-L", "-o", output_path, url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Neither wget nor curl found. Please install one of them.")
            sys.exit(1)


def extract_archive(archive_path: str, output_dir: str):
    """Extract a tar.bz2 archive."""
    print(f"Extracting: {archive_path}")
    
    try:
        subprocess.run(["tar", "xvf", archive_path, "-C", output_dir], check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to extract archive. Make sure tar is installed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download models for sherpa-onnx Voice Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to save downloaded models"
    )
    parser.add_argument(
        "--skip-vad",
        action="store_true",
        help="Skip VAD model download"
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper model download"
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS model download"
    )
    
    args = parser.parse_args()
    
    # Create models directory
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("sherpa-onnx Voice Assistant Model Downloader")
    print("="*60)
    print(f"Models directory: {models_dir.absolute()}")
    print()
    
    # 1. Download VAD model
    if not args.skip_vad:
        print("\n[1/3] Downloading Silero VAD model...")
        vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
        vad_path = models_dir / "silero_vad.onnx"
        
        if vad_path.exists():
            print(f"VAD model already exists: {vad_path}")
        else:
            download_file(vad_url, str(vad_path))
            print(f"VAD model saved to: {vad_path}")
    else:
        print("\n[1/3] Skipping VAD model download")
    
    # 2. Download Whisper tiny multilingual model
    if not args.skip_whisper:
        print("\n[2/3] Downloading Whisper tiny multilingual model...")
        whisper_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2"
        whisper_archive = models_dir / "sherpa-onnx-whisper-small.tar.bz2"
        whisper_dir = models_dir / "sherpa-onnx-whisper-small"
        
        if whisper_dir.exists():
            print(f"Whisper model already exists: {whisper_dir}")
        else:
            download_file(whisper_url, str(whisper_archive))
            extract_archive(str(whisper_archive), str(models_dir))
            
            # Remove archive after extraction
            whisper_archive.unlink()
            print(f"Whisper model saved to: {whisper_dir}")
    else:
        print("\n[2/3] Skipping Whisper model download")
    
    # 3. Download Piper TTS Hindi Rohan voice
    if not args.skip_tts:
        print("\n[3/3] Downloading Piper TTS Hindi Rohan voice...")
        tts_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-rohan-medium.tar.bz2"
        tts_archive = models_dir / "vits-piper-hi_IN-rohan-medium.tar.bz2"
        tts_dir = models_dir / "vits-piper-hi_IN-rohan-medium"
        
        if tts_dir.exists():
            print(f"TTS model already exists: {tts_dir}")
        else:
            download_file(tts_url, str(tts_archive))
            extract_archive(str(tts_archive), str(models_dir))
            
            # Remove archive after extraction
            tts_archive.unlink()
            print(f"TTS model saved to: {tts_dir}")
    else:
        print("\n[3/3] Skipping TTS model download")
    
    # Print summary
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print("\nModel files:")
    
    vad_path = models_dir / "silero_vad.onnx"
    whisper_dir = models_dir / "sherpa-onnx-whisper-tiny"
    tts_dir = models_dir / "vits-piper-hi_IN-rohan-medium"
    
    if vad_path.exists():
        print(f"  VAD: {vad_path}")
    if whisper_dir.exists():
        print(f"  Whisper: {whisper_dir}")
        print(f"    - Encoder: {whisper_dir}/tiny-encoder.int8.onnx")
        print(f"    - Decoder: {whisper_dir}/tiny-decoder.int8.onnx")
        print(f"    - Tokens: {whisper_dir}/tiny-tokens.txt")
    if tts_dir.exists():
        print(f"  TTS: {tts_dir}")
        print(f"    - Model: {tts_dir}/hi_IN-rohan-medium.onnx")
        print(f"    - Tokens: {tts_dir}/tokens.txt")
        if (tts_dir / "espeak-ng-data").exists():
            print(f"    - Data: {tts_dir}/espeak-ng-data")
    
    # Print usage example
    print("\n" + "="*60)
    print("Usage Example:")
    print("="*60)
    print(f"""
python sherpa_voice_assistant.py \\
    --silero-vad-model {vad_path} \\
    --whisper-encoder {whisper_dir}/tiny-encoder.int8.onnx \\
    --whisper-decoder {whisper_dir}/tiny-decoder.int8.onnx \\
    --whisper-tokens {whisper_dir}/tiny-tokens.txt \\
    --vits-model {tts_dir}/hi_IN-rohan-medium.onnx \\
    --vits-tokens {tts_dir}/tokens.txt \\
    --vits-data-dir {tts_dir}/espeak-ng-data
""")


if __name__ == "__main__":
    main()