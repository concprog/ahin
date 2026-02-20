import sys
import queue
import time
import multiprocessing as mp
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import os

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("Please install sounddevice: uv add sounddevice")

try:
    import soxr
except ImportError:
    raise ImportError("Please install soxr: uv add soxr")

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("Please install faster-whisper")

from .vad import VoiceActivityDetector
from .core import PiperTTSProtocol, ResponseStrategyProtocol

class VoiceAssistantFaster:
    """
    Voice Assistant using multiprocessing for intensive ASR processing.
    Audio callback runs in separate thread (efficient for I/O).
    VAD + ASR runs in separate process. Uses faster-whisper instead of pywhispercpp.
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
            'config': config,
            'needs_resampling': self.needs_resampling,
            'input_sample_rate': self.sample_rate,
            'asr_sample_rate': self.asr_sample_rate
        }
        
        self.is_running = False
        self.asr_process = None

    def _audio_callback(self, indata, frames, time_info, status):
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
        Separate process for CPU/GPU-intensive ASR work.
        """
        print(f"ASR worker started (PID: {mp.current_process().pid})")
        
        # Unpack config
        config = worker_config['config']
        needs_resampling = worker_config['needs_resampling']
        input_sample_rate = worker_config['input_sample_rate']
        asr_sample_rate = worker_config['asr_sample_rate']
        
        # Initialize ASR
        # We assume config["models"]["faster_whisper"] points to a model id or path, or default to base
        model_size_or_path = config.get("models", {}).get("faster_whisper", "small")
        n_threads = config.get("asr", {}).get("num_threads", 4)
        language = config.get("asr", {}).get("language", "hi")
        
        device = "cpu"
        compute_type = "int8"
            
        print(f"Loading faster-whisper model '{model_size_or_path}' on {device} ({compute_type})...")
        print(f"Using {n_threads} CPU threads for model ops")
        
        try:
            model = WhisperModel(
                model_size_or_path,
                device=device,
                compute_type=compute_type,
                cpu_threads=n_threads,
            )
        except Exception as e:
            print(f"Failed to load faster-whisper model: {e}")
            result_queue.put(('error', str(e)))
            return
            
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
                quality='MQ'
            )
        
        try:
            while True:
                try:
                    # Get audio chunk
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
                
                # Resample if needed
                if resampler is not None:
                    audio_data = resampler.resample_chunk(audio_data, last=False)
                
                # Feed to VAD
                vad.accept_waveform(audio_data)
                
                # Check for detected speech segments
                processed_count = 0
                while not vad.empty() and processed_count < 2:
                    segment = vad.get_speech_segment()
                    if segment is not None and len(segment) > 0:
                        # Transcribe the segment using faster-whisper
                        segments, info = model.transcribe(
                            segment,
                            language=language,
                            beam_size=5,
                            vad_filter=False,  # We already did VAD externally
                            without_timestamps=True
                        )
                        
                        text = "".join([s.text for s in segments]).strip()
                        if text:
                            result_queue.put(('transcription', text))
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
        # Generate response
        response, _ = self.response_strategy.generate_response(text)
        print(f"[Response] {response}")
        
        if _:
            # TTS
            output_path = None
            if self.config["tts"].get("output_to_file", False):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_path = str(Path.cwd() / f"{timestamp}.mp3")

            result = self.tts.synthesize(response, output_path=output_path)
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
        
        # Start ASR worker process
        self.asr_process = mp.Process(
            target=self._asr_worker,
            args=(self.audio_queue, self.result_queue, self.worker_config, self.tts_playing)
        )
        self.asr_process.start()
        
        # Result handler runs in main process
        import threading
        result_thread = threading.Thread(target=self._handle_results)
        result_thread.start()
        
        try:
            # Start Audio Stream
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
    mp.set_start_method('spawn', force=True)
