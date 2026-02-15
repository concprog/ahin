
from typing import Dict, Any
import numpy as np

try:
    import sherpa_onnx
except ImportError:
    import sys
    print("Please install sherpa-onnx: pip install sherpa-onnx")
    sys.exit(-1)


class WhisperASR:
    """Wrapper for Whisper ASR using sherpa-onnx."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Whisper ASR recognizer from configuration.
        
        Args:
            config: Configuration dictionary containing models and asr settings
        """
        models = config["models"]
        asr_cfg = config["asr"]
        
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=models["whisper_encoder"],
            decoder=models["whisper_decoder"],
            tokens=models["whisper_tokens"],
            num_threads=asr_cfg["num_threads"],
            language=asr_cfg["language"],
            task=asr_cfg["task"],
            debug=asr_cfg["debug"],
        )
        self.sample_rate = asr_cfg.get("sample_rate", 16000)
        
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples as float32 numpy array
            
        Returns:
            Transcribed text string
        """
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()
