#!/usr/bin/env python3
"""
sherpa-onnx Voice Assistant with VAD, Whisper Multilingual ASR, and Piper TTS (Rohan Voice)

This script creates a voice assistant that:
1. Uses Silero VAD for voice activity detection
2. Uses Whisper tiny multilingual for speech recognition
3. Uses Piper TTS with Hindi Rohan voice for text-to-speech
4. Optionally integrates with an LLM for assistant responses

Requirements:
- sherpa-onnx: pip install sherpa-onnx
- sounddevice: pip install sounddevice
- numpy: pip install numpy
- soundfile: pip install soundfile

Model downloads:
1. VAD: wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx 
2. Whisper tiny multilingual: wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2 
3. Piper TTS Rohan: wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-rohan-medium.tar.bz2 

Usage:
    python sherpa_voice_assistant.py

Author: Voice Assistant based on sherpa-onnx
"""

import os
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(-1)

try:
    import soundfile as sf
except ImportError:
    pass # soundfile is optional for saving files

try:
    import sherpa_onnx
except ImportError:
    print("Please install sherpa-onnx: pip install sherpa-onnx")
    sys.exit(-1)

from .config import DEFAULT_CONFIG, validate_config, merge_configs
from .core import VoiceActivityDetectorProtocol, WhisperASRProtocol, PiperTTSProtocol, ResponseStrategyProtocol
from .vad import VoiceActivityDetector
from .asr import WhisperASR
from .tts import PiperTTS


# ============================================================================
# VOICE ASSISTANT
# ============================================================================

class VoiceAssistant:
    """
    Complete voice assistant combining VAD, ASR, TTS, and a Response Strategy.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 vad: VoiceActivityDetectorProtocol,
                 asr: WhisperASRProtocol,
                 tts: PiperTTSProtocol,
                 response_strategy: ResponseStrategyProtocol):
        """
        Initialize the voice assistant.
        
        Args:
            config: Configuration dictionary
            vad: Initialized VAD instance
            asr: Initialized ASR instance
            tts: Initialized TTS instance
            response_strategy: Strategy to generate responses
        """
        self.config = config
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.response_strategy = response_strategy
        
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        
    def process_audio(self):
        """Process audio from queue for VAD and ASR."""
        buffer = []
        
        while self.is_running:
            try:
                samples = self.audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, samples]) if len(buffer) > 0 else samples
                
                # Feed to VAD in window-sized chunks
                while len(buffer) >= self.vad.window_size:
                    self.vad.accept_waveform(buffer[:self.vad.window_size])
                    buffer = buffer[self.vad.window_size:]
                
                # Process speech segments
                while not self.vad.empty():
                    speech = self.vad.get_speech_segment()
                    if speech is not None and len(speech) > 0:
                        # Transcribe
                        text = self.asr.transcribe(speech)
                        if text:
                            print(f"\n[ASR] {text}")
                            
                            # Generate response
                            response = self.response_strategy.generate_response(text)
                                
                            print(f"[Response] {response}")
                            
                            # Queue for TTS
                            self.tts_queue.put(response)
                            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                import traceback
                traceback.print_exc()
                
    def process_tts(self):
        """Process text from queue for TTS synthesis."""
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=0.1)
                
                # Synthesize and play
                result = self.tts.synthesize(text)
                if result:
                    audio, sample_rate = result
                    print(f"[TTS] Playing response...")
                    sd.play(audio, sample_rate)
                    sd.wait()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in TTS processing: {e}")
                
    def run(self):
        """Run the voice assistant with microphone input."""
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            return
            
        print("\nAvailable audio devices:")
        print(devices)
        
        default_device = sd.default.device[0]
        print(f"\nUsing device: {devices[default_device]['name']}")
        
        audio_cfg = self.config["audio"]
        sample_rate = audio_cfg["sample_rate"]
        samples_per_read = int(audio_cfg["chunk_duration"] * sample_rate)
        
        print("\n" + "="*50)
        print("Voice Assistant Started!")
        print("Speak into your microphone...")
        print("Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        self.is_running = True
        
        # Start processing threads
        audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        tts_thread = threading.Thread(target=self.process_tts, daemon=True)
        audio_thread.start()
        tts_thread.start()
        
        printed_speech = False
        
        try:
            with sd.InputStream(
                channels=audio_cfg["channels"], 
                dtype="float32", 
                samplerate=sample_rate
            ) as stream:
                while self.is_running:
                    samples, _ = stream.read(samples_per_read)
                    samples = samples.reshape(-1)
                    
                    # Put in queue for processing
                    self.audio_queue.put(samples.copy())
                    
                    # Show speech detection status
                    if self.vad.is_speech_detected() and not printed_speech:
                        print("[VAD] Speech detected...", end="", flush=True)
                        printed_speech = True
                    elif not self.vad.is_speech_detected() and printed_speech:
                        print(" ended")
                        printed_speech = False
                        
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.is_running = False
            self.vad.flush()
            
            # Wait for threads to finish
            audio_thread.join(timeout=1.0)
            tts_thread.join(timeout=1.0)
            
            print("Voice Assistant stopped.")

