# Credits and Acknowledgments

This project stands on the shoulders of giants. We are grateful to the open-source community and the creators of the following libraries and projects that made **Ahin** possible.

## Libraries and Tools

*   **`bm25s`**: Efficient BM25 sparse retrieval.
    *   [GitHub](https://github.com/dorianbrown/bm25s)
*   **`numpy`**: The fundamental package for numerical computing with Python.
    *   [Website](https://numpy.org/)
*   **`onnxruntime`**: High-performance ONNX inference engine.
    *   [GitHub](https://github.com/microsoft/onnxruntime)
*   **`openai`**: Python client library for the OpenAI API.
    *   [GitHub](https://github.com/openai/openai-python)
*   **`piper-onnx`**: A fast, local, and privacy-preserving neural text-to-speech system, used for the Piper TTS engine.
    *   [GitHub](https://github.com/rhasspy/piper)
*   **`python-dotenv`**: Reads key-value pairs from a `.env` file and sets them as environment variables.
    *   [GitHub](https://github.com/theskumar/python-dotenv)
*   **`pywhispercpp`**: Python bindings for `whisper.cpp`, a high-performance C++ implementation of OpenAI's Whisper model.
    *   [GitHub](https://github.aom/abdeladim-s/pywhispercpp)
    *   **`whisper.cpp`**: The underlying C++ library.
        *   [GitHub](https://github.com/ggerganov/whisper.cpp)
*   **`setuptools`**: Python's standard library for package management.
    *   [Website](https://setuptools.pypa.io/)
*   **`sherpa-onnx`**: A next-generation speech frontend for streaming ASR, used for VAD and potentially other speech processing tasks.
    *   [GitHub](https://github.com/k2-fsa/sherpa-onnx)
*   **`sounddevice`**: Portable audio I/O with Python.
    *   [GitHub](https://github.com/spatialaudio/python-sounddevice)
*   **`soundfile`**: Audio file I/O using Libsndfile.
    *   [GitHub](https://github.com/bastibe/python-soundfile)
*   **`soxr`**: High-quality, multithreaded sample-rate converter.
    *   [GitHub](https://github.com/chirlu/soxr)
*   **`thundersvm-cpu`**: A fast SVM library.
    *   [GitHub](https://github.com/Xtra-Computing/ThunderSVM)
*   **`tokenizers`**: Fast state-of-the-art tokenizers.
    *   [GitHub](https://github.com/huggingface/tokenizers)
*   **`webrtcvad`**: Python interface to the WebRTC Voice Activity Detector.
    *   [GitHub](https://github.com/wiseman/py-webrtcvad)

*   **Arm Kleidi Libraries**: Libraries providing optimized primitives for Arm platforms.
    *   [Arm Kleidi](https://developer.arm.com/downloads/-/arm-kleidi)
*   **Arm Compute Library**: A comprehensive set of optimized computer vision and machine learning functions for Arm CPUs and GPUs.
    *   [Arm Compute Library](https://developer.arm.com/ip-products/processors/gpu/mali-gpus/developer-resources/mali-performance-libraries/arm-compute-library)
*   **Arm NN SDK**: A machine learning inference engine that bridges existing neural network frameworks to Arm IP.
    *   [Arm NN](https://developer.arm.com/ip-products/processors/machine-learning/arm-nn)
*   **CMSIS-NN**: Efficient neural network kernels for ARM Cortex-M microcontrollers.
    *   [CMSIS-NN](https://developer.arm.com/embedded/cmsis/cmsis-nn)
*   **Arm Performance Libraries**: Optimized standard numerical libraries for HPC on Arm.
    *   [Arm Performance Libraries](https://developer.arm.com/tools-and-software/server-and-hpc/arm-performance-libraries)

## Models

The following models are integral to the functionality of Ahin:

*   **Silero VAD Model**: Used for Voice Activity Detection, typically managed via `sherpa-onnx`.
    *   [Source (often part of `sherpa-onnx` releases or k2-fsa repositories)](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)
*   **Whisper Model (GGML based)**: A compact version of OpenAI's Whisper model, used for Automatic Speech Recognition (ASR) via `pywhispercpp`. Specifically, `ggml` quantized models are used for performance on ARM.
    *   [Original Whisper Model](https://github.com/openai/whisper)
    *   [GGML models often found in `whisper.cpp` releases or community repositories](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
*   **Piper TTS Hindi Rohan Voice**: A high-quality neural text-to-speech voice for Hindi, used by `piper-onnx`.
    *   [Source (often part of `piper-onnx` releases or community repositories)](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-rohan-medium.tar.bz2)
