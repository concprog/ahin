# Usage and Configuration

This document provides an overview of the project's Python files and details on how to configure the voice assistant via `ahin/config.py`.

## Project Files Overview

Here's a brief description of the main Python files in the project:

*   **`main.py`**: This is the primary entry point for the Ahin voice assistant application. It initializes the core components, loads the configuration, and starts the assistant's main loop.
*   **`download_models.py`**: A utility script responsible for fetching and preparing the various ONNX-based models required by the voice assistant, including models for Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS).
*   **`ahin/`**: This directory constitutes the core Python package for Ahin.
    *   **`ahin/__init__.py`**: Marks the `ahin` directory as a Python package.
    *   **`ahin/asr.py`**: (Note: This file is currently not used by `main.py`'s `VoiceAssistantFast` implementation, which uses `pywhispercpp` directly for ASR.) This file would typically contain logic related to Automatic Speech Recognition.
    *   **`ahin/config.py`**: Defines the default configuration settings for the entire voice assistant, covering model paths, VAD parameters, ASR options, TTS settings, and audio I/O.
    *   **`ahin/core.py`**: Contains fundamental interfaces (Python Protocols) and base classes used across the project, such as `PiperTTSProtocol` and `ResponseStrategyProtocol`, ensuring consistent component interactions.
    *   **`ahin/tts.py`**: Implements the Text-to-Speech (TTS) functionality, primarily interacting with the Piper TTS engine.
    *   **`ahin/vad.py`**: Handles Voice Activity Detection (VAD) by wrapping the `sherpa-onnx` VAD implementation to identify speech segments in the audio stream.
    *   **`ahin/voice_assistant_fast.py`**: The high-performance implementation of the voice assistant. It utilizes multiprocessing to run CPU-intensive ASR tasks in a separate process, optimizing for real-time responsiveness on resource-constrained hardware like the Raspberry Pi.
    *   **`ahin/voice_assistant.py`**: (Note: This file is currently not used by `main.py` and is likely an older or alternative implementation of the voice assistant.)
    *   **`ahin/strats/`**: This subdirectory contains various response strategies the assistant can employ.
        *   **`ahin/strats/__init__.py`**: Marks the `strats` directory as a Python package.
        *   **`ahin/strats/conversational.py`**: Implements a conversational response strategy based on predefined pattern matching for common Hindi phrases.
        *   **`ahin/strats/default.py`**: Provides a simple default response strategy, primarily echoing back the user's input.
        *   **`ahin/strats/llm.py`**: Implements a response strategy that leverages a Large Language Model (LLM) for generating more dynamic and intelligent responses.
        *   **`ahin/strats/semantic.py`**: (Note: This file is currently a placeholder for a semantic search-based response strategy.)

## Configuration Settings in `ahin/config.py`

The `ahin/config.py` file defines a `DEFAULT_CONFIG` dictionary that governs the behavior of the voice assistant. You can customize these settings without directly modifying `ahin/config.py` by editing the `create_custom_config` function within your `main.py` file. This function allows you to override specific default values.

Here's a breakdown of the main sections in `DEFAULT_CONFIG` and how to adjust them:

### `models`

This section specifies the file paths to all the necessary ONNX and `ggml` models.

```python
"models": {
    "vad": "./models/silero_vad.onnx",
    "whisper_encoder": "./models/sherpa-onnx-whisper-small/small-encoder.onnx",
    "whisper_decoder": "./models/sherpa-onnx-whisper-small/small-decoder.onnx",
    "whisper_tokens": "./models/sherpa-onnx-whisper-small/small-tokens.txt",
    "whisper_cpp": "./models/ggml-base-hi.bin", # Used by VoiceAssistantFast
    "vits_model": "./models/vits-piper-hi_IN-rohan-medium/hi_IN-rohan-medium.onnx",
    "vits_config": "./models/vits-piper-hi_IN-rohan-medium/hi_IN-rohan-medium.onnx.json",
    "vits_tokens": "./models/vits-piper-hi_IN-rohan-medium/tokens.txt",
    "vits_data_dir": "./models/vits-piper-hi_IN-rohan-medium/espeak-ng-data",
},
```

*   **`vad`**: Path to the Silero VAD ONNX model.
*   **`whisper_encoder`, `whisper_decoder`, `whisper_tokens`**: Paths for `sherpa-onnx` based Whisper models (not used by `VoiceAssistantFast` for ASR).
*   **`whisper_cpp`**: Path to the `ggml` Whisper model used by `pywhispercpp` in `VoiceAssistantFast`. **This is the critical path for ASR model when using `VoiceAssistantFast`.**
*   **`vits_model`, `vits_config`, `vits_tokens`, `vits_data_dir`**: Paths for the Piper TTS VITS model and its associated files.

**To change a model path:**
Modify `main.py`'s `create_custom_config` function:
```python
def create_custom_config() -> Dict[str, Any]:
    custom = {
        "models": {
            "whisper_cpp": "/path/to/your/new-ggml-model.bin",
            "vits_model": "/path/to/your/new-piper-model.onnx",
        }
    }
    return custom
```

### `vad` (Voice Activity Detection)

Parameters controlling the VAD's behavior.

```python
"vad": {
    "sample_rate": 16000,
    "buffer_size_seconds": 20,
    "min_silence_duration": 0.5,
    "min_speech_duration": 0.1,
    "threshold": 0.6,
},
```

*   **`sample_rate`**: Expected audio sample rate for VAD.
*   **`buffer_size_seconds`**: How much audio the VAD buffers.
*   **`min_silence_duration`**: Minimum duration of silence to consider speech ended.
*   **`min_speech_duration`**: Minimum duration of detected speech to be considered valid.
*   **`threshold`**: VAD sensitivity (higher value means less sensitive to speech).

**To adjust VAD settings:**
```python
def create_custom_config() -> Dict[str, Any]:
    custom = {
        "vad": {
            "min_silence_duration": 1.0, # Increase silence duration
            "threshold": 0.7, # Make VAD less sensitive
        }
    }
    return custom
```

### `asr` (Automatic Speech Recognition)

Settings for the ASR engine.

```python
"asr": {
    "language": "hi",  # Empty for auto-detect, or "hi", "en", "zh", etc.
    "task": "transcribe",  # or "translate"
    "num_threads": 4,
    "debug": False,
    "sample_rate": 16000,
},
```

*   **`language`**: Target language for transcription (e.g., "hi" for Hindi, "en" for English). Can be empty for auto-detection.
*   **`task`**: "transcribe" for speech-to-text, or "translate" for speech-to-English translation.
*   **`num_threads`**: Number of CPU threads to use for ASR inference.
*   **`debug`**: Enable/disable ASR debugging output.
*   **`sample_rate`**: Expected audio sample rate for ASR.

**To change ASR settings:**
```python
def create_custom_config() -> Dict[str, Any]:
    custom = {
        "asr": {
            "language": "en", # Change to English ASR
            "num_threads": 2, # Use fewer threads
        }
    }
    return custom
```

### `tts` (Text-to-Speech)

Parameters for the TTS engine.

```python
"tts": {
    "num_threads": 4,
    "speed": 1.1,
    "debug": False,
    "sample_rate": 22050,
    "output_to_file": False,
},
```

*   **`num_threads`**: Number of CPU threads to use for TTS synthesis.
*   **`speed`**: Playback speed of the synthesized speech (e.g., 1.0 is normal, 1.5 is 50% faster).
*   **`debug`**: Enable/disable TTS debugging output.
*   **`sample_rate`**: Output sample rate of the synthesized audio.
*   **`output_to_file`**: If `True`, synthesized speech will also be saved to MP3 files in the current working directory.

**To adjust TTS settings:**
```python
def create_custom_config() -> Dict[str, Any]:
    custom = {
        "tts": {
            "speed": 0.9, # Slower speech
            "output_to_file": True, # Save audio to file
        }
    }
    return custom
```

### `audio` (Audio I/O)

Settings related to microphone input and speaker output.

```python
"audio": {
    "sample_rate": 48000,
    "chunk_duration": 0.1,  # 100ms chunks
    "channels": 1,
},
```

*   **`sample_rate`**: The sample rate of the audio hardware (microphone/speaker).
*   **`chunk_duration`**: The size of audio chunks processed at a time, in seconds.
*   **`channels`**: Number of audio channels (typically 1 for mono).

**To change audio I/O settings:**
```python
def create_custom_config() -> Dict[str, Any]:
    custom = {
        "audio": {
            "sample_rate": 16000, # If your hardware prefers 16kHz
            "chunk_duration": 0.05, # Smaller chunks for lower latency
        }
    }
    return custom
```

### `assistant` (Assistant Behavior)

General behavior settings for the assistant.

```python
"assistant": {
    "response_language": "hindi",  # For default responses
}
```

*   **`response_language`**: The language in which the assistant generates its default or fallback responses.

**To change assistant behavior:**
```python
def create_custom_config() -> Dict[str, Any]:
    custom = {
        "assistant": {
            "response_language": "english", # Assistant responds in English
        }
    }
    return custom
```
