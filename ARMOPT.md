# ARM Optimization & Hardware Tuning

This document details the specific optimization strategies employed to run **Ahin**, a fully local Hindi voice assistant, on the constrained hardware of a Raspberry Pi 4. Our goal was to achieve near-real-time latency (sub-second response) without relying on cloud APIs.

## Hardware Platform: Raspberry Pi 4 Model B

The Raspberry Pi 4 is a capable single-board computer, but it significantly lags behind modern mobile devices in raw compute power, especially for AI workloads.

### Specifications
*   **SoC:** Broadcom BCM2711
*   **CPU:** Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ **1.5GHz** (Stock)
*   **GPU:** VideoCore VI 3D Graphics
*   **RAM:** 4GB / 8GB LPDDR4-3200 SDRAM
*   **Storage:** MicroSD (High bottleneck for I/O) / USB 3.0 SSD (Recommended)
*   **AI Accelerators:** None (No NPU/TPU)

### Comparison vs. Standard Android Smartphone
Comparing the RPi 4 to a standard mid-range Android phone (e.g., Pixel 7a or Samsung S21 FE):

| Feature | Raspberry Pi 4 | Standard Android Phone (Mid-Range) | Impact |
| :--- | :--- | :--- | :--- |
| **Architecture** | 4x High-Perf Cores (Old A72) | 8x Cores (Mix of X1/A78/A55) | Phone has much better burst performance. |
| **Clock Speed** | 1.5 GHz | 2.4 GHz - 2.8 GHz+ | Phone is ~2x faster per clock. |
| **AI Hardware** | **None** (CPU Only) | **DSP / NPU / TPU** | Phones offload AI; Pi must burn CPU cycles. |
| **Thermal** | Passive (Throttles easily) | Passive (Designed for bursts) | Pi requires active cooling for sustained AI. |
| **Instruction Set**| ARMv8 (NEON) | ARMv8/v9 (NEON + DotProd) | Newer ARM chips have better matrix math support. |

**Challenge:** We needed to make a heavy AI pipeline (VAD -> Speech-to-Text -> LLM/Logic -> Text-to-Speech) run smoothly on a device with a fraction of a phone's power and no dedicated AI hardware.

---

## Hardware Tuning: Overclocking

To squeeze every bit of performance out of the Cortex-A72 cores, we applied significant overclocking. This directly improves the inference speed of the Whisper and VITS models.

**Warning:** This requires active cooling (fan/heatsink) and a high-quality power supply.

**`/boot/firmware/config.txt` Settings:**

```ini
# Overclock to 2.1 GHz (requires active cooling)
over_voltage=6
arm_freq=2147

# GPU Overclock (helps UI/Display, minor impact on compute)
gpu_freq=750
```

*   **Result:** ~30-40% improvement in token generation speed and transcription time compared to stock 1.5GHz.

---

## GPU Acceleration Experiments: Vulkan & VideoCore VI

We conducted extensive experiments attempting to utilize the Raspberry Pi 4's integrated **VideoCore VI GPU** for compute acceleration using the **Vulkan** API. Our hypothesis was that offloading the heavy matrix multiplications required by the Whisper encoder to the GPU would free up the CPU for other tasks. We compiled `sherpa-onnx` and tested `pywhispercpp` configurations with Vulkan compute support.

However, the reality of the hardware architecture proved different from our expectations. The VideoCore VI is primarily designed for 3D graphics (OpenGL ES) rather than general-purpose compute (GPGPU). The driver overhead introduced by the Mesa V3D stack added significant initialization latency. Furthermore, the GPU lacks the raw GFLOPS to compete with the quad-core CPU when the CPU is running highly optimized instructions. The shared memory architecture also meant that contention for memory bandwidth between the CPU and GPU created new bottlenecks. Consequently, we determined that **highly optimized CPU inference (Int8 Quantized) was significantly faster** than our GPU acceleration attempts for this specific workload. The overhead of moving data to and from the weak GPU outweighed the potential compute benefits.

---

## Software Optimization: The "Fast" Pipeline

Since hardware acceleration was limited, we focused on extreme software optimization.

### 1. Voice Activity Detection (VAD)

For detecting when a user is speaking, we chose **Silero VAD via ONNX Runtime**. While `webrtcvad` is faster, it is less robust to noise, and the standard PyTorch implementation of Silero is too heavy. By using the `.onnx` export of the Silero model, we configured the specialized `sherpa-onnx` runtime to leverage the **ARM NN ONNX runtime** for acceleration. This allowed us to process audio chunks in milliseconds, incurring negligible CPU usage and keeping the cores free for the heavy lifting required by the ASR engine.

### 2. Automatic Speech Recognition (ASR) with pywhispercpp & NEON

The core of our speech recognition engine is built upon **`pywhispercpp`**, a Python binding for the highly optimized C++ port of OpenAI's Whisper model (`whisper.cpp`). Standard PyTorch implementations of Whisper are often too heavy for the Raspberry Pi's limited memory bandwidth and lack of hardware acceleration. The Python Global Interpreter Lock (GIL) also makes it difficult to achieve real-time performance with pure Python implementations.

We compiled `pywhispercpp` specifically to leverage the **ARM NEON** instruction set available on the Cortex-A72. NEON provides Single Instruction Multiple Data (SIMD) capabilities, allowing the processor to perform the same mathematical operation on multiple data points simultaneously. This is crucial for the dense matrix multiplications inherent in neural network inference. By bypassing Python's slowness and running the heavy inference in pure C++ with NEON SIMD instructions, we achieve a dramatic speedup.

We further optimized the model by employing **8-bit integer (Int8) quantization**. This reduces the precision of the model weights from 32-bit floating point to 8-bit integers. While this theoretically reduces accuracy, the impact on speech recognition quality is negligible for our use case. However, the performance gains are massive: it reduces memory usage by approximately 4x and significantly increases memory throughput, which is often the primary bottleneck on the Raspberry Pi 4. We also configured the decoder to use greedy decoding (disabling beam search) to favor raw speed over marginally better accuracy.

### 3. Text-to-Speech (TTS)

For speech synthesis, we selected **Piper TTS (ONNX)**. Unlike Google TTS which requires an internet connection and introduces network latency, or Tacotron2 which is too computationally expensive, Piper uses the VITS architecture exported to ONNX. It is designed specifically for low-resource devices like the Steam Deck or Raspberry Pi. It generates high-quality Hindi speech locally with less than 200ms latency on our overclocked hardware.

### 4. System Architecture: Multithreading & Multiprocessing

The Python Global Interpreter Lock (GIL) poses a significant challenge for real-time audio applications, as it prevents multiple native threads from executing Python bytecodes simultaneously. To circumvent this, we architected Ahin using a strict **multiprocess design**. The main process is dedicated entirely to lightweight, latency-sensitive I/O operations: it manages the microphone input stream, handles the speaker output, and updates the UI. This ensures that no matter how heavy the computational load becomes, the audio buffer never overflows or underflows, preventing the dreaded 'stuttering' audio.

Heavy computational tasks—specifically the Voice Activity Detection (VAD) and Automatic Speech Recognition (ASR)—are offloaded to a **separate, isolated worker process**. We utilize `multiprocessing.Queue` as an Inter-Process Communication (IPC) mechanism to pass raw audio frames from the main process to the worker. This producer-consumer model allows the ASR engine to utilize the full power of the CPU cores without blocking the main event loop. While the worker crunches data using `pywhispercpp`, the main process remains responsive, listening for the next user input or handling cancellation requests. This architecture allows us to achieve true parallelism on the quad-core CPU.

### 5. Consideration of ARM-Specific AI/ML & HPC Libraries

In our quest for optimal performance on the ARM architecture, we extensively explored and considered various specialized libraries designed to accelerate AI, ML, and high-performance computing (HPC) workloads. Our investigation included cross-framework tools such as the **Arm Kleidi Libraries**, which provide low-level, optimized primitives for various computations, and the more comprehensive **Arm Compute Library**, offering a rich set of accelerated functions for machine learning, computer vision, and signal processing. We also evaluated the **Arm NN SDK**, a middleware designed to bridge popular machine learning frameworks like TensorFlow and PyTorch with highly optimized ARM backends, often leveraging the Compute Library or dedicated NPUs (though the RPi4 lacks an NPU). For more embedded-centric applications, **CMSIS-NN** was on our radar for its highly optimized kernels for common neural network layers, though its primary focus is typically microcontrollers rather than application processors like the Cortex-A72. Furthermore, in the realm of high-performance computing, we looked into **Arm Performance Libraries** for potential acceleration of any custom numerical routines that might arise. The final selection of our core AI components (Sherpa-ONNX, pywhispercpp, Piper TTS) ultimately provided the best balance of performance and ease of integration for our specific use case, often internally leveraging the very types of optimizations these libraries offer.

## Summary of Latency Gains

| Component | Unoptimized (Stock Pi 4) | Optimized (OC + C++ + ONNX) |
| :--- | :--- | :--- |
| **VAD** | 50ms | **<10ms** |
| **Transcribe (3s Audio)** | 4.5s (PyTorch) | **~0.8s (WhisperCPP Int8)** |
| **TTS Generation** | 2.0s | **0.3s (Piper)** |
| **Total Response Time** | ~7-8s | **~1.5s** |

By carefully selecting packages that use **C++ backends** and **ONNX runtimes**, and pushing the hardware to its thermal limits, we transformed the Raspberry Pi 4 from a sluggish device into a capable, offline Hindi voice assistant.