import argparse
import sounddevice as sd
from engines import AudioRecorder, QwenASR, QwenTTS, QwenLLM

# System prompt for LLM
SYSTEM_PROMPT = "あなたはユーザーの友人です。気軽に、簡潔に回答してください。"

# LLM parameters (passed to model.generate())
LLM_GENERATE_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "do_sample": True,
}

# ASR/TTS parameters
LANGUAGE = "Japanese"
TTS_LANGUAGE = "Japanese"


def main():
    parser = argparse.ArgumentParser(description="Voice chat with Qwen")
    parser.add_argument("--voice", required=True, help="Voice preset name")
    args = parser.parse_args()

    print("Loading ASR model...")
    asr = QwenASR()

    print("Loading TTS model...")
    tts = QwenTTS(voice_name=args.voice)

    print("Loading LLM model...")
    llm = QwenLLM()

    recorder = AudioRecorder()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\nSystem ready. Let's talk!")

    try:
        while True:
            print("\n[Input] Press Enter to start recording (Ctrl+C to exit)...")
            try:
                input()
            except KeyboardInterrupt:
                break

            recorder.start()
            print("Recording... Press Enter to stop.")

            try:
                input()
            except KeyboardInterrupt:
                break

            audio = recorder.stop()
            if audio.size == 0:
                continue

            print("Transcribing...")
            user_text = asr.transcribe(audio, language=LANGUAGE)
            if not user_text.strip():
                print("Could not hear anything.")
                continue

            print(f"\n[You]: {user_text}")

            print("Thinking...")
            messages.append({"role": "user", "content": user_text})
            response_text = llm.chat(messages, **LLM_GENERATE_KWARGS)
            messages.append({"role": "assistant", "content": response_text})

            print(f"[Qwen]: {response_text}")

            print("Synthesizing...")
            success, _ = tts.synthesize(response_text, language=TTS_LANGUAGE)

            if not success:
                print("Failed to produce audio response.")

    except KeyboardInterrupt:
        print("\nExiting voice chat...")
    finally:
        print("Goodbye!")

if __name__ == "__main__":
    main()
