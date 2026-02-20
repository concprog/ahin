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

We attempted to utilize the Raspberry Pi 4's integrated **VideoCore VI GPU** for compute acceleration using the **Vulkan** API provided by `sherpa-onnx` and `ncnn`.

### The Experiment
We compiled `sherpa-onnx` with Vulkan compute support to offload the matrix multiplications required by the Whisper encoder.

### The Reality
While technically functional, the VideoCore VI is primarily designed for 3D graphics (OpenGL ES), not general-purpose compute (GPGPU).
*   **Driver Overhead:** The Mesa V3D driver stack added significant initialization latency.
*   **Compute Power:** The GPU lacks the raw GFLOPS to compete with the quad-core CPU running highly optimized NEON SIMD instructions.
*   **Memory Bandwidth:** Shared memory contention between CPU and GPU created bottlenecks.

**Conclusion:** For the Raspberry Pi 4 specifically, **highly optimized CPU inference (Int8 Quantized) proved faster than GPU acceleration.** The overhead of moving data to the weak GPU outweighed the compute benefits.

---

## Software Optimization: The "Fast" Pipeline

Since hardware acceleration was limited, we focused on extreme software optimization.

### 1. Voice Activity Detection (VAD)
*   **Standard Approach:** `webrtcvad` (Fast but less accurate) or PyTorch Silero (Heavy).
*   **Our Choice:** **Silero VAD via ONNX Runtime**.
*   **Why:** We use the `.onnx` export of Silero. Running this on the specialized `sherpa-onnx` runtime allows us to process audio chunks in milliseconds with negligible CPU usage, keeping the cores free for the heavy lifting (ASR).

### 2. Automatic Speech Recognition (ASR)
*   **Standard Approach:** OpenAI `whisper` (Python/PyTorch).
*   **Bottleneck:** PyTorch is too heavy for RPi 4; Python GIL blocks real-time audio processing.
*   **Our Choice:** **`pywhispercpp`** (Python bindings for `whisper.cpp`).
*   **Optimization:**
    *   **C++ Core:** The heavy inference runs in pure C++, bypassing Python's slowness.
    *   **Quantization:** We use the `tiny` or `small` models quantized to **Int8**. This reduces memory usage by 4x and utilizes ARM NEON instructions for rapid inference.
    *   **Greedy Decoding:** We disabled beam search (num_beams=1) to favor speed over marginally better accuracy.

### 3. Text-to-Speech (TTS)
*   **Standard Approach:** Google TTS (Online/Slow) or Tacotron2 (Too heavy).
*   **Our Choice:** **Piper TTS (ONNX)**.
*   **Why:** Piper uses the VITS architecture exported to ONNX. It is designed specifically for low-resource devices (like the Steam Deck or Pi). It generates high-quality Hindi speech locally with <200ms latency on an overclocked Pi 4.

### 4. System Architecture: Multiprocessing
To prevent audio stuttering while the CPU works on transcription:
*   **Process 1 (Main):** Handles Audio I/O (Microphone/Speaker) and lightweight Logic.
*   **Process 2 (Worker):** Dedicated isolated process for the ASR (Whisper) engine.
*   **IPC:** We use `multiprocessing.Queue` to pass raw audio frames. This ensures that even if the CPU is 100% utilized transcribing the last sentence, the microphone recording loop in the main process never drops a frame.

## Summary of Latency Gains

| Component | Unoptimized (Stock Pi 4) | Optimized (OC + C++ + ONNX) |
| :--- | :--- | :--- |
| **VAD** | 50ms | **<10ms** |
| **Transcribe (3s Audio)** | 4.5s (PyTorch) | **~0.8s (WhisperCPP Int8)** |
| **TTS Generation** | 2.0s | **0.3s (Piper)** |
| **Total Response Time** | ~7-8s | **~1.5s** |

By carefully selecting packages that use **C++ backends** and **ONNX runtimes**, and pushing the hardware to its thermal limits, we transformed the Raspberry Pi 4 from a sluggish device into a capable, offline Hindi voice assistant.
