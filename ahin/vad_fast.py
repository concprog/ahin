# vad_onnxruntime.py

from typing import Dict, Any, Optional
from collections import deque
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    import sys
    print("Please install onnxruntime: pip install onnxruntime")
    sys.exit(-1)


# ---------------------------------------------------------------------------
# Shared internal state machine that mirrors Silero VAD logic
# ---------------------------------------------------------------------------

class _SileroVADState:
    """
    Runs Silero VAD inference and manages the speech/silence state machine.
    Compatible with silero-vad v4 (single-LSTM, window=512 @ 16kHz).
    """

    WINDOW_SIZE = 512          # samples per inference step at 16 kHz

    def __init__(
        self,
        session: ort.InferenceSession,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration: float = 0.3,
        min_speech_duration: float = 0.1,
        buffer_size_seconds: float = 30.0,
    ):
        self._session = session
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_samples = int(min_silence_duration * sample_rate)
        self.min_speech_samples  = int(min_speech_duration  * sample_rate)
        self.buffer_size_samples = int(buffer_size_seconds  * sample_rate)

        # LSTM hidden / cell state  (shape expected by Silero v4: 2×1×64)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array([sample_rate], dtype=np.int64)

        # Internal ring buffer for incoming audio
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)

        # Speech segment accumulator and output queue
        self._speech_buf: np.ndarray = np.empty(0, dtype=np.float32)
        self._segments: deque = deque()

        # State machine
        self._in_speech = False
        self._silence_samples = 0

    # ------------------------------------------------------------------
    # Public interface (mirrors sherpa-onnx VoiceActivityDetector)
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        return self.WINDOW_SIZE

    def accept_waveform(self, samples: np.ndarray):
        samples = np.asarray(samples, dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, samples])
        self._process_buffer()

    def flush(self):
        """Pad remaining buffer to window size and process."""
        remainder = len(self._buffer)
        if remainder > 0:
            pad = self.WINDOW_SIZE - (remainder % self.WINDOW_SIZE)
            if pad != self.WINDOW_SIZE:
                self._buffer = np.concatenate(
                    [self._buffer, np.zeros(pad, dtype=np.float32)]
                )
            self._process_buffer()
        # Finalise any open speech segment
        if self._in_speech and len(self._speech_buf) >= self.min_speech_samples:
            self._segments.append(self._speech_buf.copy())
        self._speech_buf = np.empty(0, dtype=np.float32)
        self._in_speech = False

    def is_speech_detected(self) -> bool:
        return self._in_speech

    def empty(self) -> bool:
        return len(self._segments) == 0

    def front_samples(self) -> Optional[np.ndarray]:
        return self._segments[0] if self._segments else None

    def pop(self):
        if self._segments:
            self._segments.popleft()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(self, window: np.ndarray) -> float:
        """Run one window through the ONNX model and return speech probability."""
        inp = window.reshape(1, self.WINDOW_SIZE)
        outputs = self._session.run(
            None,
            {
                "input":       inp,
                "h":           self._h,
                "c":           self._c,
                "sr":          self._sr,
            },
        )
        # Silero v4 output order: output, hn, cn
        prob, self._h, self._c = outputs[0], outputs[1], outputs[2]
        return float(prob.squeeze())

    def _process_buffer(self):
        while len(self._buffer) >= self.WINDOW_SIZE:
            window = self._buffer[: self.WINDOW_SIZE]
            self._buffer = self._buffer[self.WINDOW_SIZE :]
            prob = self._infer(window)
            self._update_state(window, prob)

    def _update_state(self, window: np.ndarray, prob: float):
        if prob >= self.threshold:
            # Speech frame
            self._silence_samples = 0
            self._speech_buf = np.concatenate([self._speech_buf, window])
            if not self._in_speech:
                self._in_speech = True
        else:
            # Silence frame
            if self._in_speech:
                self._speech_buf = np.concatenate([self._speech_buf, window])
                self._silence_samples += self.WINDOW_SIZE
                if self._silence_samples >= self.min_silence_samples:
                    # End of speech segment
                    if len(self._speech_buf) >= self.min_speech_samples:
                        self._segments.append(self._speech_buf.copy())
                    self._speech_buf = np.empty(0, dtype=np.float32)
                    self._in_speech = False
                    self._silence_samples = 0


# ---------------------------------------------------------------------------
# Helper: build an onnxruntime session with a given list of providers
# ---------------------------------------------------------------------------

def _make_session(model_path: str, providers: list) -> ort.InferenceSession:
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)


# ===========================================================================
# 1.  XNNPACK Execution Provider
# ===========================================================================

class VoiceActivityDetectorXNNPACK:
    """
    Silero VAD using onnxruntime with the XNNPACK execution provider.

    XNNPACK is a highly-optimised inference library for ARM/x86 CPUs.
    It is included in the standard `onnxruntime` wheel on most platforms.

    Install:
        pip install onnxruntime          # CPU wheel (includes XNNPACK on supported builds)
        # or for a build that explicitly enables XNNPACK:
        pip install onnxruntime-extensions   # optional

    Note: XNNPACK EP support depends on your onnxruntime build.
    If unavailable the session will silently fall back to CPUExecutionProvider.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Same schema as the sherpa-onnx wrapper:
                config["models"]["vad"]   – path to silero_vad.onnx
                config["vad"]["sample_rate"]
                config["vad"]["buffer_size_seconds"]
                config["vad"]["min_silence_duration"]
                config["vad"]["min_speech_duration"]
                config["vad"]["threshold"]
        """
        model_path = config["models"]["vad"]
        vad_cfg    = config["vad"]

        providers = [
            (
                "XNNPACKExecutionProvider",
                {
                    "intra_op_num_threads": vad_cfg.get("num_threads", 1),
                },
            ),
            "CPUExecutionProvider",   # fallback
        ]

        session = _make_session(model_path, providers)

        self._state = _SileroVADState(
            session=session,
            sample_rate=vad_cfg["sample_rate"],
            threshold=vad_cfg["threshold"],
            min_silence_duration=vad_cfg["min_silence_duration"],
            min_speech_duration=vad_cfg["min_speech_duration"],
            buffer_size_seconds=vad_cfg["buffer_size_seconds"],
        )
        self.sample_rate = vad_cfg["sample_rate"]
        self.window_size = self._state.window_size

    # ------------------------------------------------------------------
    # Public API – identical to the sherpa-onnx wrapper
    # ------------------------------------------------------------------

    def accept_waveform(self, samples: np.ndarray):
        """Add audio samples to the VAD buffer."""
        self._state.accept_waveform(samples)

    def is_speech_detected(self) -> bool:
        """True while a speech segment is open."""
        return self._state.is_speech_detected()

    def empty(self) -> bool:
        """True when there are no completed speech segments queued."""
        return self._state.empty()

    def get_speech_segment(self) -> Optional[np.ndarray]:
        """Pop and return the next completed speech segment, or None."""
        samples = self._state.front_samples()
        if samples is not None:
            self._state.pop()
            return samples
        return None

    def flush(self):
        """Flush remaining audio and finalise any open speech segment."""
        self._state.flush()


# ===========================================================================
# 2.  ARMNN Execution Provider
# ===========================================================================

class VoiceActivityDetectorARMNN:
    """
    Silero VAD using onnxruntime with the ARMNN execution provider.

    ARMNN is Arm's ML inference framework, optimised for Cortex-A / Mali-G
    devices (Raspberry Pi 4/5, Jetson, i.MX 8, etc.).

    Install (Arm devices only):
        pip install onnxruntime          # base wheel
        # ARMNN EP requires a custom onnxruntime build that includes ARMNN.
        # Official Arm builds:  https://github.com/ARM-software/armnn
        # or pre-built wheels:  pip install ort-armnn  (community package)

    Provider options (all optional, shown with defaults):
        config["vad"]["armnn_compute_library"]  – "CpuAcc" | "GpuAcc" (default "CpuAcc")
        config["vad"]["armnn_fast_math"]        – True | False         (default True)
        config["vad"]["num_threads"]            – int                  (default 1)
    """

    def __init__(self, config: Dict[str, Any]):
        model_path = config["models"]["vad"]
        vad_cfg    = config["vad"]

        compute_lib = vad_cfg.get("armnn_compute_library", "CpuAcc")
        fast_math   = vad_cfg.get("armnn_fast_math", True)

        providers = [
            (
                "ArmNNExecutionProvider",
                {
                    # "CpuAcc" uses Arm Compute Library on the CPU (NEON / SVE).
                    # "GpuAcc" offloads to a Mali GPU via OpenCL.
                    "compute_library": compute_lib,
                    "fast_math_enabled": str(fast_math).lower(),
                },
            ),
            "CPUExecutionProvider",   # fallback if ARMNN EP is unavailable
        ]

        session = _make_session(model_path, providers)

        self._state = _SileroVADState(
            session=session,
            sample_rate=vad_cfg["sample_rate"],
            threshold=vad_cfg["threshold"],
            min_silence_duration=vad_cfg["min_silence_duration"],
            min_speech_duration=vad_cfg["min_speech_duration"],
            buffer_size_seconds=vad_cfg["buffer_size_seconds"],
        )
        self.sample_rate = vad_cfg["sample_rate"]
        self.window_size = self._state.window_size

    # ------------------------------------------------------------------
    # Public API – identical to the sherpa-onnx wrapper
    # ------------------------------------------------------------------

    def accept_waveform(self, samples: np.ndarray):
        self._state.accept_waveform(samples)

    def is_speech_detected(self) -> bool:
        return self._state.is_speech_detected()

    def empty(self) -> bool:
        return self._state.empty()

    def get_speech_segment(self) -> Optional[np.ndarray]:
        samples = self._state.front_samples()
        if samples is not None:
            self._state.pop()
            return samples
        return None

    def flush(self):
        self._state.flush()