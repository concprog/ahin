#!/usr/bin/env python3
import sys
from typing import Dict, Any

from ahin.config import DEFAULT_CONFIG, validate_config, merge_configs
from ahin.voice_assistant import VoiceAssistant
from ahin.vad import VoiceActivityDetector
from ahin.asr import WhisperASR
from ahin.tts import PiperTTS
from ahin.strats.default import DefaultResponseStrategy


def create_custom_config() -> Dict[str, Any]:
    """
    Create a custom configuration by modifying defaults.
    Override this function to customize settings without modifying DEFAULT_CONFIG.
    """
    custom = {
        # Example: Override model paths
        # "models": {
        #     "vad": "/path/to/your/silero_vad.onnx",
        #     "whisper_encoder": "/path/to/your/encoder.onnx",
        #     ...
        # },
        
        # Example: Force Hindi language detection
        "asr": {
            "language": "hi",
        },
        
        # Example: Speed up speech
        # "tts": {
        #     "speed": 1.2,
        # },
    }
    return custom


def main():
    """Main entry point."""
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Apply custom overrides
    custom = create_custom_config()
    config = merge_configs(config, custom)
    
    # Validate configuration
    print("Validating configuration...")
    if not validate_config(config):
        print("\nPlease ensure all model files are downloaded.")
        sys.exit(1)
        
    print("Initializing Voice Assistant...")
    print("="*50)
    
    # Initialize components
    try:
        vad = VoiceActivityDetector(config)
        asr = WhisperASR(config)
        tts = PiperTTS(config)
        response_strategy = DefaultResponseStrategy(config)
        
        from ahin.voice_assistant_fast import VoiceAssistantFast
        assistant = VoiceAssistantFast(config, tts, response_strategy)
        assistant.run()
    except Exception as e:
        print(f"Error initializing components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()