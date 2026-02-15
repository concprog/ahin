
import sys
import queue
import time
import threading
import logging
from typing import Dict, Any, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(-1)

from pywhispercpp.model import Model
from .vad import VoiceActivityDetector
from .core import PiperTTSProtocol, ResponseStrategyProtocol

class VoiceAssistantFast:
    """
    Voice Assistant using pywhispercpp for ASR and local VAD (Sherpa-ONNX via vad.py).
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
        
        # Audio Configuration
        self.sample_rate = 16000
        self.block_size = int(self.sample_rate * 0.03) # 30ms block
        
        # VAD
        self.vad = VoiceActivityDetector(config)
        self.audio_queue = queue.Queue()
        
        # ASR
        # Using a default GGML model path or one from config if it existed.
        # The user's previous code used various defaults. We'll stick to a reasonable default.
        model_path = config.get("models", {}).get("whisper_cpp", "./models/ggml-base-hi.bin")
        n_threads = config.get("asr", {}).get("num_threads", 4)
        
        print(f"Loading pywhispercpp model from {model_path}...")
        self.asr_model = Model(model_path,
                               n_threads=n_threads,
                               print_realtime=False,
                               print_progress=False,
                               print_timestamps=False,
                               single_segment=True,
                               no_context=True)
                               
        self.is_running = False

    def _audio_callback(self, indata, frames, time, status):
        """
        Audio callback for sounddevice.
        """
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        
        # Add to queue for processing in the main loop/thread
        # We make a copy to avoid buffer reuse issues
        self.audio_queue.put(indata.copy())

    def _process_audio(self):
        """
        Process audio from queue: VAD -> Accumulate -> Transcribe
        """
        print("Listening...")
        
        while self.is_running:
            try:
                # Get audio chunk
                # Using a timeout to allow checking self.is_running occasionally
                indata = self.audio_queue.get(timeout=0.1)
                
                # Flatten and process for VAD
                # vad.py accepts numpy array. Sherpa VAD expects flat array.
                audio_data = indata.flatten()
                
                # Feed to VAD
                self.vad.accept_waveform(audio_data)
                
                # Check for detected speech segments
                while not self.vad.empty():
                    segment = self.vad.get_speech_segment()
                    if segment is not None and len(segment) > 0:
                        self._transcribe_segment(segment)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")

    def _transcribe_segment(self, audio_segment):
        """
        Transcribe a speech segment using pywhispercpp.
        """
        # pywhispercpp expects float array.
        # Transcribe
        try:
            # We can pass the segment directly
            self.asr_model.transcribe(audio_segment, new_segment_callback=self._on_transcription)
        except Exception as e:
            print(f"Transcription error: {e}")

    def _on_transcription(self, seg):
        """Callback for new transcription segments."""
        text = seg.text.strip()
        if text:
            print(f"\n[ASR] {text}")
            self._handle_command(text)

    def _handle_command(self, text: str):
        """Generate response and speak."""
        # Generate response
        response = self.response_strategy.generate_response(text)
        print(f"[Response] {response}")
        
        if response:
            # TTS
            # Note: This blocks the processing loop, which is actually good 
            # because we don't want to listen to our own voice.
            # However, sounddevice stream is still running in background and determining VAD.
            # We might want to assume the user IS listening to the TTS and pause processing?
            # But the VAD is robust enough usually.
            
            # Simple approach: Stop processing queue while speaking?
            # Or just let it run. If we have echo cancellation it's fine. 
            # Without it, we might trigger ourselves.
            # Let's simple play.
            
            result = self.tts.synthesize(response)
            if result:
                audio, sample_rate = result
                print(f"[TTS] Playing...")
                sd.play(audio, sample_rate)
                sd.wait()
                # Clear queue after speaking to avoid processing eco
                with self.audio_queue.mutex:
                    self.audio_queue.queue.clear()
                self.vad.flush()

    def run(self):
        """Start the assistant."""
        self.is_running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.start()
        
        try:
            # Start Audio Stream
            with sd.InputStream(channels=1,
                                samplerate=self.sample_rate,
                                blocksize=self.block_size,
                                callback=self._audio_callback):
                print("Press Ctrl+C to stop...")
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.is_running = False
            process_thread.join()
            print("Stopped.")
