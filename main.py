#!/usr/bin/env python3
import sys
import time
from typing import Dict, Any

from ahin.config import DEFAULT_CONFIG, validate_config, merge_configs
from dotenv import load_dotenv

load_dotenv()
# from ahin.voice_assistant import VoiceAssistant # Deprecated/Unused
# from ahin.vad import VoiceActivityDetector # Used internally by VoiceAssistantFast
# from ahin.asr import WhisperASR # Deprecated/Unused
# from ahin.tts import PiperTTS
from ahin.tts import PiperOnnxTTS
from ahin.voice_assistant_fast import VoiceAssistantFast


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
        "tts": {
            "speed": 1.25,
        },
        
        # LLM Configuration
        "llm": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "nvidia/nemotron-3-nano-30b-a3b",
            # API Key is loaded from environment variable NVIDIA_API_KEY
        }
    }
    return custom


def main():
    """Main entry point."""
    init_start = time.perf_counter()
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Apply custom overrides
    custom = create_custom_config()
    config = merge_configs(config, custom)
    
    # Validate configuration
    print("Validating configuration...")
    validate_start = time.perf_counter()
    if not validate_config(config):
        print("\nPlease ensure all model files are downloaded.")
        sys.exit(1)
    validate_time = time.perf_counter() - validate_start
    print(f"⏱️  Config validation: {validate_time*1000:.1f}ms")
        
    print("Initializing Voice Assistant...")
    print("="*50)
    
    # Initialize components
    try:
        # Initialize TTS
        tts_start = time.perf_counter()
        # tts = PiperTTS(config)
        tts = PiperOnnxTTS(config)
        tts_time = time.perf_counter() - tts_start
        print(f"⏱️  TTS initialization: {tts_time*1000:.1f}ms")
        
        # Initialize Response Strategy
        strategy_start = time.perf_counter()
        # Use LLM Strategy
        from ahin.strats.llm import ConversationalStrategy
        response_strategy = ConversationalStrategy(config)
        strategy_time = time.perf_counter() - strategy_start
        print(f"⏱️  Strategy initialization: {strategy_time*1000:.1f}ms")
        
        # Initialize Voice Assistant
        assistant_start = time.perf_counter()
        from ahin.voice_assistant_fast import VoiceAssistantFast
        # Using VoiceAssistantFast as requested
        # Note: VAD (Sherpa-ONNX) and ASR (pywhispercpp) are handled internally
        assistant = VoiceAssistantFast(config, tts, response_strategy)
        assistant_time = time.perf_counter() - assistant_start
        print(f"⏱️  Assistant initialization: {assistant_time*1000:.1f}ms")
        
        total_init = time.perf_counter() - init_start
        print(f"⏱️  TOTAL initialization: {total_init*1000:.1f}ms")
        print("="*50)
        
        assistant.run()
    except Exception as e:
        print(f"Error initializing components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()