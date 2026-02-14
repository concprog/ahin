
import sys
from typing import Dict, Any, Optional
import threading
import queue

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(-1)

import numpy as np
from pywhispercpp.examples.assistant import Assistant

from .core import PiperTTSProtocol, ResponseStrategyProtocol

class VoiceAssistantFast:
    """
    Voice Assistant using pywhispercpp for faster VAD and ASR.
    Retains the same ResponseStrategy and TTS pipeline.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 tts: PiperTTSProtocol,
                 response_strategy: ResponseStrategyProtocol):
        """
        Initialize the fast voice assistant.
        
        Args:
            config: Configuration dictionary
            tts: Initialized TTS instance
            response_strategy: Strategy to generate responses
        """
        self.config = config
        self.tts = tts
        self.response_strategy = response_strategy
        self.assistant = None
        self.tts_queue = queue.Queue()
        self.is_running = False
        
    def _command_callback(self, text: str):
        """Callback for pywhispercpp Assistant when speech is transcribed."""
        if not text.strip():
            return
            
        print(f"\n[ASR] {text}")
        
        # Generate response
        response = self.response_strategy.generate_response(text)
        print(f"[Response] {response}")
        
        # Synthesize and play (we can do this directly or via queue)
        # Using queue to keep callback fast is better, but pywhispercpp might be running in a thread already.
        # Let's try direct execution first, or use the existing TTS loop pattern if we want to be safe.
        # Given pywhispercpp runs in its own loops, we should probably output audio in a way that doesn't block capture too much,
        # but since we are replying, blocking capture is actually desired (to avoid hearing itself).
        
        # Synthesize
        result = self.tts.synthesize(response)
        if result:
            audio, sample_rate = result
            print(f"[TTS] Playing response...")
            # Blocking playback to prevent the assistant from listening to itself
            sd.play(audio, sample_rate)
            sd.wait()

    def run(self):
        """Run the pywhispercpp assistant."""
        print("Initializing pywhispercpp Assistant...")
        
        # Extract settings from config if available, otherwise use defaults
        # Note: pywhispercpp Assistant config is passed via init args
        model_name = "tiny.en" # Default, maybe map from config if possible, but pywhispercpp models are different names
        
        # We can try to map config['asr']['language'] to a model if we want, 
        # but usually pywhispercpp uses model names like 'base', 'small', etc.
        # For now, we'll stick to a default or let user configure it via a specific key if we added one.
        # Let's use 'base' as a reasonable default for better accuracy than tiny.
        
        self.assistant = Assistant(
            commands_callback=self._command_callback,
            n_threads=self.config.get("asr", {}).get("num_threads", 4),
            model="small", # hardcoded for better quality than tiny, matches our previous config intent roughly
            # input_device=... # We could map this if we knew the ID
            silence_threshold=16, # Default
            block_duration=30     # Default
        )
        
        print("\n" + "="*50)
        print("Fast Voice Assistant Started (pywhispercpp)!")
        print("Speak into your microphone...")
        print("Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        self.assistant.start()
