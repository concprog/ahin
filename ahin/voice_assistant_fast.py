import sys
import queue
import time
import multiprocessing as mp
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("Please install sounddevice: uv add sounddevice")

try:
    import soxr
except ImportError:
    raise ImportError("Please install soxr: uv add soxr")

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
        print("<-- PYWHISPERCPP + SHERPA-ONNX Voice Assistant -->")
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
        self.audio_queue = mp.Queue(maxsize=100)  # Limit queue size to prevent memory bloat
        self.result_queue = mp.Queue()
        self.tts_playing = mp.Event()
        
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
    def _asr_worker(audio_queue: mp.Queue, result_queue: mp.Queue, worker_config: dict, tts_playing: Any):
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
                     no_context=True,
                     audio_ctx=512,
                     language=config.get("asr", {}).get("language", "hi")
                     )
        
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
                quality='MQ'  # High quality - fast enough for real-time
            )
        
        def transcribe_callback(seg):
            """Callback for transcription results."""
            text = seg.text.strip()
            if text:
                result_queue.put(('transcription', text))
        
        try:
            while True:
                try:
                    # Get audio chunk (wait up to 1 second before looping)
                    indata = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if indata is None:  # Poison pill to stop
                    break
                
                # Check if TTS is playing - skip processing
                if tts_playing.is_set():
                    continue
                
                # Flatten
                audio_data = indata.flatten()
                
                # Resample if needed (input_rate -> 16kHz)
                if resampler is not None:
                    resample_start = time.perf_counter()
                    audio_data = resampler.resample_chunk(audio_data, last=False)
                    resample_time = time.perf_counter() - resample_start
                    # Only print occasionally to avoid spam (every 100 chunks)
                    if hasattr(resampler, '_debug_count'):
                        resampler._debug_count += 1
                    else:
                        resampler._debug_count = 1
                    if resampler._debug_count % 100 == 0:
                        print(f"⏱️  [Worker] Resample: {resample_time*1000:.2f}ms")
                
                # Feed to VAD (now at 16kHz if resampled)
                vad_start = time.perf_counter()
                vad.accept_waveform(audio_data)
                vad_time = time.perf_counter() - vad_start
                
                # Check for detected speech segments
                processed_count = 0
                while not vad.empty() and processed_count < 2:
                    segment_start = time.perf_counter()
                    segment = vad.get_speech_segment()
                    segment_time = time.perf_counter() - segment_start
                    
                    if segment is not None and len(segment) > 0:
                        segment_duration = len(segment) / asr_sample_rate
                        print(f"⏱️  [Worker] VAD detected speech: {segment_duration:.2f}s segment, extraction: {segment_time*1000:.1f}ms")
                        
                        # Transcribe (CPU-intensive) - audio segment is already at 16kHz
                        asr_start = time.perf_counter()
                        model.transcribe(segment, new_segment_callback=transcribe_callback)
                    processed_count += 1
                        
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
        total_start = time.perf_counter()
        
        # Generate response
        response_start = time.perf_counter()
        matched, response = self.response_strategy.generate_response(text)
        response_time = time.perf_counter() - response_start
        
        print(f"[Response] {response}")
        print(f"⏱️  Response generation: {response_time*1000:.1f}ms (matched: {matched})")
        
        if response:
            # TTS
            output_path = None
            if self.config["tts"].get("output_to_file", False):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_path = str(Path.cwd() / f"{timestamp}.mp3")

            tts_start = time.perf_counter()
            result = self.tts.synthesize(response, output_path=output_path)
            tts_time = time.perf_counter() - tts_start
            print(f"⏱️  TTS synthesis: {tts_time*1000:.1f}ms")
            
            if result:
                audio, sample_rate = result
                print(f"[TTS] Playing...")
                
                # Pause audio processing during TTS to avoid echo
                self.tts_playing.set()
                
                def _play_and_clear():
                    sd.play(audio, sample_rate)
                    sd.wait()
                    time.sleep(0.2)
                    self.tts_playing.clear()
                    
                import threading
                threading.Thread(target=_play_and_clear).start()

    def run(self):
        """Start the assistant."""
        self.is_running = True
        
        # Start ASR worker process (true parallelism)
        self.asr_process = mp.Process(
            target=self._asr_worker,
            args=(self.audio_queue, self.result_queue, self.worker_config, self.tts_playing)
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
                                dtype='float32',
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