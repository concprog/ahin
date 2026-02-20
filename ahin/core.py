# Objectives: 1 decorator for strats - plugins
import functools
from typing import Annotated, Protocol, Optional, Any, Tuple
import numpy as np


def singleton(cls):
    """Make a class a Singleton class (only one instance)"""
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if wrapper_singleton.instance is None:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance
    wrapper_singleton.instance = None
    return wrapper_singleton


class ResponseStrategyProtocol(Protocol):
    def generate_response(self, text: str) -> Tuple[bool, str]:
        """
        Generate a response based on the input text.
        
        Returns:
            Tuple of (matched: bool, response: str)
            - matched: True if the strategy successfully matched and handled the input
            - response: The generated response text
        """
        ...


class VoiceActivityDetectorProtocol(Protocol):
    def accept_waveform(self, samples: np.ndarray) -> None: ...
    def is_speech_detected(self) -> bool: ...
    def empty(self) -> bool: ...
    def get_speech_segment(self) -> Optional[np.ndarray]: ...
    def flush(self) -> None: ...
    @property
    def window_size(self) -> int: ...


class WhisperASRProtocol(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class PiperTTSProtocol(Protocol):
    def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]: ...