
from typing import Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf
import sys
import time

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
        synth_start = time.perf_counter()
        audio = self.tts.generate(text, sid=0, speed=self.speed)
        synth_time = time.perf_counter() - synth_start
        
        if len(audio.samples) == 0:
            print("TTS synthesis failed")
            return None
        
        audio_duration = len(audio.samples) / audio.sample_rate
        rtf = synth_time / audio_duration if audio_duration > 0 else 0
        print(f"⏱️  [TTS Sherpa] Synthesis: {synth_time*1000:.1f}ms for {audio_duration:.2f}s audio (RTF: {rtf:.2f}x)")
            
        if output_path:
            # We don't want to fail if soundfile is not available or path is invalid, 
            # but we should try to save if requested
            try:
                save_start = time.perf_counter()
                sf.write(output_path, audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
                save_time = time.perf_counter() - save_start
                print(f"Saved TTS audio to {output_path} ({save_time*1000:.1f}ms)")
            except Exception as e:
                print(f"Error saving TTS audio: {e}")
            
        return np.array(audio.samples, dtype=np.float32), audio.sample_rate


Piper: Optional[Any] = None
try:
    from piper_onnx import Piper
except ImportError:
    print("For PiperOnnxTTS, please install piper-onnx: uv add piper-onnx")
    # We do not exit here, as the user might not be using this class


class PiperOnnxTTS:
    """Wrapper for Piper TTS using piper-onnx (official python binding)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Piper TTS synthesizer from configuration.
        
        Args:
            config: Configuration dictionary containing models and tts settings
        """
        models = config["models"]
        
        # Check if piper-onnx is installed
        if 'piper_onnx' not in sys.modules and 'Piper' not in globals():
             raise ImportError("piper-onnx is not installed. Please install it with: uv add piper-onnx")

        # Load model and config
        model_path = models["vits_model"]
        config_path = models.get("vits_config")
        
        if not config_path:
             # Try to infer config path if not provided
             config_path = model_path + ".json"
             
        self.piper = Piper(model_path, config_path)
        self.speed = 1.0 # Piper-onnx does not seem to support speed adjustment in create() directly in the provided snippet
        # If the user wants speed adjustment, it might need post-processing or checking piper-onnx docs.
        # However, the user request snippet didn't show speed adjustment.
        
    def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save the audio file
            
        Returns:
            Tuple of (audio_samples, sample_rate) or None if synthesis failed
        """
        # The user example: samples, sample_rate = piper.create('Hello world from Piper!', speaker_id=voices['awb'])
        # But we might not need speaker_id if it's a single speaker model.
        # Let's check voices first.
        # voices = self.piper.get_voices()
        # If voices is empty or None, maybe we don't pass speaker_id.
        # The example used speaker_id=voices['awb'].
        
        # For single speaker models, speaker_id might be Optional.
        # Let's try to call it without speaker_id first, or check if we need it.
        # Actually, the user snippet showed:
        # voices = piper.get_voices()
        # samples, sample_rate = piper.create('Hello world from Piper!', speaker_id=voices['awb'])
        
        # I'll implement a safe way to get speaker_id.
        speaker_id: Optional[int] = None
        voices = self.piper.get_voices()
        if voices:
            # creating a default speaker id if available.
            # config.json usually has speaker_id_map.
            # converting to list and taking first one if detailed map.
            # But the 'voices' return from piper-onnx seems to be a dict or list.
            # In the user snippet: `speaker_id=voices['awb']`.
            # So voices is a dict.
            # I will just pick the first one if available.
            try:
                 first_voice_key = next(iter(voices))
                 speaker_id = voices[first_voice_key]
            except StopIteration:
                 pass
        
        text = text.strip().replace("\n", " ")
        if not text:
            return None

        # Piper onnx create returns (samples, sample_rate)
        
        # We need to satisfy the linter for speaker_id which expects int or str usually, but python binding might accept None?
        # If speaker_id is None, let's assume the library handles it or we should pass 0?
        # For now we cast to Any to avoid lint error if we are sure it works or just leave it.
        # But wait, looking at the snippet `create('Hello world from Piper!', speaker_id=voices['awb'])`
        
        synth_start = time.perf_counter()
        if speaker_id is None:
             samples, sample_rate = self.piper.create(text)
        else:
             samples, sample_rate = self.piper.create(text, speaker_id=speaker_id)
        synth_time = time.perf_counter() - synth_start
        
        # Convert to float32 for compatibility with sounddevice if it's int16
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        audio_duration = len(samples) / sample_rate
        rtf = synth_time / audio_duration if audio_duration > 0 else 0
        print(f"⏱️  [TTS Piper-ONNX] Synthesis: {synth_time*1000:.1f}ms for {audio_duration:.2f}s audio (RTF: {rtf:.2f}x)")

        if output_path:
            try:
                # Save using soundfile
                save_start = time.perf_counter()
                sf.write(output_path, samples, samplerate=sample_rate)
                save_time = time.perf_counter() - save_start
                print(f"Saved Piper-ONNX audio to {output_path} ({save_time*1000:.1f}ms)")
            except Exception as e:
                print(f"Error saving TTS audio: {e}")
                
        return samples, sample_rate
