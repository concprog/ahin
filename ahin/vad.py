
from typing import Dict, Any, Optional
import numpy as np

try:
    import sherpa_onnx
except ImportError:
    import sys
    print("Please install sherpa-onnx: pip install sherpa-onnx")
    sys.exit(-1)


class VoiceActivityDetector:
    """Wrapper for Silero VAD using sherpa-onnx."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VAD from configuration dictionary.
        
        Args:
            config: Configuration dict with keys: model_path, sample_rate, 
                   buffer_size_seconds, min_silence_duration, min_speech_duration, threshold
        """
        model_path = config["models"]["vad"]
        vad_cfg = config["vad"]
        
        self.config = sherpa_onnx.VadModelConfig()
        self.config.silero_vad.model = model_path
        self.config.silero_vad.min_silence_duration = vad_cfg["min_silence_duration"]
        self.config.silero_vad.min_speech_duration = vad_cfg["min_speech_duration"]
        self.config.silero_vad.threshold = vad_cfg["threshold"]
        self.config.sample_rate = vad_cfg["sample_rate"]
        
        self.sample_rate = vad_cfg["sample_rate"]
        # Creating a temporary VAD to access default window size if needed, or rely on internal defaults
        self._vad = sherpa_onnx.VoiceActivityDetector(
            self.config, buffer_size_in_seconds=vad_cfg["buffer_size_seconds"]
        )
        self.window_size = self.config.silero_vad.window_size
        
    def accept_waveform(self, samples: np.ndarray):
        """Add audio samples to the VAD buffer."""
        self._vad.accept_waveform(samples)
        
    def is_speech_detected(self) -> bool:
        """Check if speech is currently detected."""
        return self._vad.is_speech_detected()
    
    def empty(self) -> bool:
        """Check if there are no speech segments in the queue."""
        return self._vad.empty()
    
    def get_speech_segment(self) -> Optional[np.ndarray]:
        """Get the next speech segment from the queue."""
        if not self.empty():
            samples = self._vad.front.samples
            self._vad.pop()
            return np.array(samples, dtype=np.float32)
        return None
    
    def flush(self):
        """Flush any remaining audio in the buffer."""
        self._vad.flush()
