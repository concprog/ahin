# Future Enhancements and Quality Improvements

This document outlines potential future directions for enhancing the performance and, more critically, the overall quality of the Ahin voice assistant. Our goal is to achieve an even more natural, accurate, and responsive conversational experience while maintaining local, on-device processing.

## 1. ASR Quality and Speed Improvements

While `pywhispercpp` provides excellent performance on the Raspberry Pi 4, there's always room for improvement in accuracy and speed, especially for a Hinglish (Hindi + English code-switching) context.

*   **Model Distillation for Hinglish ASR:** We aim to explore the distillation of larger, more accurate Hinglish ASR models, such as the [ShunyaLabs Zero-shot STT Hinglish model](https://huggingface.co/shunyalabs/zero-stt-hinglish), into smaller, faster versions suitable for on-device inference. This process involves training a smaller "student" model to mimic the behavior of a larger "teacher" model, thereby retaining much of the accuracy at a significantly reduced computational cost. Such a specialized model could greatly enhance understanding of mixed-language input.
*   **Fine-tuning Whisper:** Experimenting with fine-tuning smaller Whisper models (e.g., `tiny` or `base`) on custom Hinglish datasets to improve domain-specific or accent-specific recognition.
*   **Beam Search Optimization:** Re-evaluating beam search (currently disabled for speed) with optimized implementations or adaptive strategies that balance latency and accuracy.

## 2. TTS Naturalness and Expressiveness

The Piper TTS engine delivers good performance, but further improvements can be made to make the generated speech sound more human-like.

*   **Custom Voice Models:** Training or integrating custom Piper TTS voice models specifically tailored for desired speaking styles or additional Hindi dialects.
*   **Emotion and Prosody Control:** Investigating methods to control the emotion, intonation, and prosody of the synthesized speech, making responses more empathetic and natural.

## 3. Advanced Response Strategies (LLM & Semantic)

The current LLM strategy offers dynamic responses, but its quality can be significantly enhanced.

*   **Smaller, Specialized LLMs:** Researching and integrating smaller, highly optimized LLMs that can run efficiently on the Raspberry Pi for more complex conversational turns.
*   **Intent Recognition and Slot Filling:** Implementing a dedicated intent recognition module to accurately understand user commands and extract relevant information (slot filling) for more robust control of smart home features or other actions.
*   **Semantic Search Integration:** Developing the placeholder semantic search strategy (`ahin/strats/semantic.py`) to allow the assistant to retrieve information from a local knowledge base.

## 4. Enhanced User Experience

*   **Wake Word Detection:** Implementing a local, low-power wake word detection system to allow the assistant to be invoked hands-free without constantly streaming audio for ASR.
*   **Visual Feedback:** Exploring integration with a display to provide visual feedback (e.g., "Listening...", "Thinking...", transcribed text).
*   **Offline Knowledge Base:** Integrating a compact, offline knowledge base for answering common queries without internet connectivity.
*   **Multilingual Support:** Expanding the assistant to support additional Indian languages (Bengali, Tamil, Telugu, Kannada, Malayalam) alongside Hindi and English, enabling broader accessibility across India's diverse linguistic landscape.
