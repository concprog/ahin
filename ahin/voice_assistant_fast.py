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

try:
    import soxr
except ImportError:
    print("Please install soxr: pip install soxr")
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
        self.input_sample_rate = config["audio"].get("sample_rate", 16000)
        self.asr_sample_rate = 16000  # ASR always requires 16kHz
        
        # Determine if resampling is needed
        self.needs_resampling = self.input_sample_rate != self.asr_sample_rate
        
        if self.needs_resampling:
            print(f"Audio resampling enabled: {self.input_sample_rate}Hz -> {self.asr_sample_rate}Hz")
            # Initialize streaming resampler for real-time processing
            self.resampler = soxr.ResampleStream(
                self.input_sample_rate,
                self.asr_sample_rate,
                1,  # mono channel
                dtype='float32',
                quality='HQ'  # High quality - fast enough for real-time
            )
        else:
            self.resampler = None
        
        self.block_size = int(self.input_sample_rate * 0.03)  # 30ms block at input rate
        
        # VAD - operates at ASR sample rate
        self.vad = VoiceActivityDetector(config)
        self.audio_queue = queue.Queue()
        
        # ASR
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

    def _resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Resample audio from input sample rate to ASR sample rate using SoXR.
        
        Args:
            audio_data: Input audio array at input_sample_rate
            
        Returns:
            Resampled audio array at asr_sample_rate
        """
        if not self.needs_resampling:
            return audio_data
        
        # Ensure float32 dtype for soxr
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Use streaming resampler for real-time processing
        resampled = self.resampler.resample_chunk(audio_data, last=False)
        
        return resampled

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
        Process audio from queue: Resample -> VAD -> Accumulate -> Transcribe
        """
        print("Listening...")
        
        while self.is_running:
            try:
                # Get audio chunk
                # Using a timeout to allow checking self.is_running occasionally
                indata = self.audio_queue.get(timeout=0.1)
                
                # Flatten
                audio_data = indata.flatten()
                
                # Resample if needed (input_rate -> 16kHz)
                audio_data = self._resample_audio(audio_data)
                
                # Feed to VAD (now at 16kHz)
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
        
        Args:
            audio_segment: Audio at 16kHz (ASR sample rate)
        """
        try:
            # Audio segment is already at 16kHz from VAD output
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
            result = self.tts.synthesize(response)
            if result:
                audio, sample_rate = result
                print(f"[TTS] Playing...")
                sd.play(audio, sample_rate)
                sd.wait()
                # Clear queue after speaking to avoid processing echo
                with self.audio_queue.mutex:
                    self.audio_queue.queue.clear()
                self.vad.flush()
                # Reset resampler state if using streaming resampler
                if self.resampler is not None:
                    self.resampler.clear()

    def run(self):
        """Start the assistant."""
        self.is_running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.start()
        
        try:
            # Start Audio Stream at input sample rate
            with sd.InputStream(channels=1,
                                samplerate=self.input_sample_rate,
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