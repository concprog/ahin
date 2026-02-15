import sys
import queue
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
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
    Voice Assistant using multiprocessing for CPU-intensive ASR processing.
    Audio callback runs in separate thread (efficient for I/O).
    VAD + ASR runs in separate process (true parallelism).
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
        
        # Audio Configuration - kept from original
        self.sample_rate = config["audio"].get("sample_rate", 16000)
        self.asr_sample_rate = 16000  # ASR always requires 16kHz
        self.block_size = int(self.sample_rate * 0.03)  # 30ms block
        
        # Determine if resampling is needed
        self.needs_resampling = self.sample_rate != self.asr_sample_rate
        
        if self.needs_resampling:
            print(f"Audio resampling enabled: {self.sample_rate}Hz -> {self.asr_sample_rate}Hz")
        
        # Multiprocessing queues (thread-safe and process-safe)
        self.audio_queue = Queue(maxsize=100)  # Limit queue size to prevent memory bloat
        self.result_queue = Queue()
        
        # Keep all original config intact for passing to worker
        self.worker_config = {
            'config': config,  # Pass entire config to worker
            'needs_resampling': self.needs_resampling,
            'input_sample_rate': self.sample_rate,
            'asr_sample_rate': self.asr_sample_rate
        }
        
        self.is_running = False
        self.asr_process = None

    def _audio_callback(self, indata, frames, time, status):
        """
        Audio callback for sounddevice - runs in audio thread.
        This is I/O bound, so threading is efficient here.
        """
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        
        try:
            # Non-blocking put with immediate drop if queue full
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # Drop frame if queue full (better than blocking)
            pass

    @staticmethod
    def _asr_worker(audio_queue: Queue, result_queue: Queue, worker_config: dict):
        """
        Separate process for CPU-intensive ASR work.
        This achieves true parallelism by bypassing GIL.
        """
        print(f"ASR worker started (PID: {mp.current_process().pid})")
        
        # Unpack config
        config = worker_config['config']
        needs_resampling = worker_config['needs_resampling']
        input_sample_rate = worker_config['input_sample_rate']
        asr_sample_rate = worker_config['asr_sample_rate']
        
        # Initialize ASR - using original config logic
        model_path = config.get("models", {}).get("whisper_cpp", "./models/ggml-base-hi.bin")
        n_threads = config.get("asr", {}).get("num_threads", 4)
        
        print(f"Loading pywhispercpp model from {model_path}...")
        model = Model(model_path,
                     n_threads=n_threads,
                     print_realtime=False,
                     print_progress=False,
                     print_timestamps=False,
                     single_segment=True,
                     no_context=True)
        
        # Initialize VAD in worker process
        from .vad import VoiceActivityDetector
        vad = VoiceActivityDetector(config)
        
        # Initialize resampler if needed
        resampler = None
        if needs_resampling:
            resampler = soxr.ResampleStream(
                input_sample_rate,
                asr_sample_rate,
                1,  # mono channel
                dtype='float32',
                quality='HQ'  # High quality - fast enough for real-time
            )
        
        def transcribe_callback(seg):
            """Callback for transcription results."""
            text = seg.text.strip()
            if text:
                result_queue.put(('transcription', text))
        
        try:
            while True:
                # Get audio chunk
                indata = audio_queue.get()
                
                if indata is None:  # Poison pill to stop
                    break
                
                # Flatten
                audio_data = indata.flatten()
                
                # Resample if needed (input_rate -> 16kHz)
                if resampler is not None:
                    # Ensure float32 dtype for soxr
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    audio_data = resampler.resample_chunk(audio_data, last=False)
                
                # Feed to VAD (now at 16kHz if resampled)
                vad.accept_waveform(audio_data)
                
                # Check for detected speech segments
                while not vad.empty():
                    segment = vad.get_speech_segment()
                    if segment is not None and len(segment) > 0:
                        # Transcribe (CPU-intensive) - audio segment is already at 16kHz
                        model.transcribe(segment, new_segment_callback=transcribe_callback)
                        
        except Exception as e:
            print(f"Error in ASR worker: {e}")
            result_queue.put(('error', str(e)))
        
        print("ASR worker stopped")

    def _handle_results(self):
        """
        Handle transcription results in main process.
        Runs TTS and response generation.
        """
        while self.is_running:
            try:
                msg_type, data = self.result_queue.get(timeout=0.1)
                
                if msg_type == 'transcription':
                    print(f"\n[ASR] {data}")
                    self._handle_command(data)
                elif msg_type == 'error':
                    print(f"[Error] {data}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error handling results: {e}")

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
                
                # Pause audio processing during TTS to avoid echo
                # Drain the audio queue
                try:
                    while True:
                        self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                
                sd.play(audio, sample_rate)
                sd.wait()

    def run(self):
        """Start the assistant."""
        self.is_running = True
        
        # Start ASR worker process (true parallelism)
        self.asr_process = Process(
            target=self._asr_worker,
            args=(self.audio_queue, self.result_queue, self.worker_config)
        )
        self.asr_process.start()
        
        # Result handler runs in main process (using threading for I/O)
        import threading
        result_thread = threading.Thread(target=self._handle_results)
        result_thread.start()
        
        try:
            # Start Audio Stream at input sample rate (callback runs in audio thread)
            with sd.InputStream(channels=1,
                                samplerate=self.sample_rate,
                                blocksize=self.block_size,
                                callback=self._audio_callback):
                print("Press Ctrl+C to stop...")
                print(f"Main process PID: {mp.current_process().pid}")
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.is_running = False
            
            # Send poison pill to ASR worker
            self.audio_queue.put(None)
            
            # Wait for processes/threads
            if self.asr_process:
                self.asr_process.join(timeout=2)
                if self.asr_process.is_alive():
                    self.asr_process.terminate()
            
            result_thread.join(timeout=2)
            print("Stopped.")


if __name__ == '__main__':
    # Required for multiprocessing on Windows/macOS
    mp.set_start_method('spawn', force=True)