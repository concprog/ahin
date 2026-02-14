
from typing import Dict, Any, List, Tuple
from pathlib import Path

# ============================================================================
# CONFIGURATION DICTIONARIES
# ============================================================================

DEFAULT_CONFIG = {
    # Model paths - adjust these to your actual model locations
    "models": {
        "vad": "./models/silero_vad.onnx",
        # "whisper_encoder": "./models/sherpa-onnx-whisper-small/small-encoder.onnx",
        # "whisper_decoder": "./models/sherpa-onnx-whisper-small/small-decoder.onnx",
        # "whisper_tokens": "./models/sherpa-onnx-whisper-small/small-tokens.txt",
        "whisper_encoder": "./models/sherpa-onnx-whisper-tiny/encoder_model_int8.onnx",
        "whisper_decoder": "./models/sherpa-onnx-whisper-tiny/decoder_model_int8.onnx",
        "whisper_tokens": "./models/sherpa-onnx-whisper-tiny/hindi-vocab-tokens.txt",
        "vits_model": "./models/vits-piper-hi_IN-rohan-medium/hi_IN-rohan-medium.onnx",
        "vits_tokens": "./models/vits-piper-hi_IN-rohan-medium/tokens.txt",
        "vits_data_dir": "./models/vits-piper-hi_IN-rohan-medium/espeak-ng-data",
    },
    
    # VAD configuration
    "vad": {
        "sample_rate": 16000,
        "buffer_size_seconds": 20,
        "min_silence_duration": 0.5,
        "min_speech_duration": 0.25,
        "threshold": 0.4,
    },
    
    # ASR configuration
    "asr": {
        "language": "hi",  # Empty for auto-detect, or "hi", "en", "zh", etc.
        "task": "transcribe",  # or "translate"
        "num_threads": 4,
        "debug": False,
    },
    
    # TTS configuration
    "tts": {
        "num_threads": 4,
        "speed": 1.1,
        "debug": False,
    },
    
    # Audio I/O configuration
    "audio": {
        "sample_rate": 16000,
        "chunk_duration": 0.1,  # 100ms chunks
        "channels": 1,
    },
    
    # Assistant behavior
    "assistant": {
        "response_language": "hindi",  # For default responses
    }
}


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate that all required model files exist."""
    required_files: List[Tuple[str, str]] = [
        (config["models"]["vad"], "VAD model"),
        (config["models"]["whisper_encoder"], "Whisper encoder"),
        (config["models"]["whisper_decoder"], "Whisper decoder"),
        (config["models"]["whisper_tokens"], "Whisper tokens"),
        (config["models"]["vits_model"], "VITS model"),
        (config["models"]["vits_tokens"], "VITS tokens"),
    ]
    
    all_valid = True
    for filepath, name in required_files:
        if not Path(filepath).is_file():
            print(f"Error: {name} not found: {filepath}")
            all_valid = False
            
    data_dir = config["models"].get("vits_data_dir", "")
    if data_dir and not Path(data_dir).is_dir():
        print(f"Warning: VITS data directory not found: {data_dir}")
        
    return all_valid


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
