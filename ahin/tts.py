
from typing import Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    import sys
    print("Please install sherpa-onnx: pip install sherpa-onnx")
    sys.exit(-1)


class PiperTTS:
    """Wrapper for Piper TTS using sherpa-onnx with Rohan voice."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Piper TTS synthesizer from configuration.
        
        Args:
            config: Configuration dictionary containing models and tts settings
        """
        models = config["models"]
        tts_cfg = config["tts"]
        
        self.speed = tts_cfg["speed"]
        self.sample_rate = tts_cfg.get("sample_rate", 22050)  # Piper default sample rate
        
        self.config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=models["vits_model"],
                    tokens=models["vits_tokens"],
                    data_dir=models.get("vits_data_dir", ""),
                    lexicon="",  # Not needed for Piper models with espeak-ng-data
                ),
                provider="cpu",
                debug=tts_cfg["debug"],
                num_threads=tts_cfg["num_threads"],
            ),
            max_num_sentences=1,
        )
        
        if not self.config.validate():
            raise ValueError("Invalid TTS configuration")
            
        self.tts = sherpa_onnx.OfflineTts(self.config)
        
    def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save the audio file
            
        Returns:
            Tuple of (audio_samples, sample_rate) or None if synthesis failed
        """
        audio = self.tts.generate(text, sid=0, speed=self.speed)
        
        if len(audio.samples) == 0:
            print("TTS synthesis failed")
            return None
            
        if output_path:
            # We don't want to fail if soundfile is not available or path is invalid, 
            # but we should try to save if requested
            try:
                sf.write(output_path, audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
                print(f"Saved TTS audio to {output_path}")
            except Exception as e:
                print(f"Error saving TTS audio: {e}")
            
        return np.array(audio.samples, dtype=np.float32), audio.sample_rate
