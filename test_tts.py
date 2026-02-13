import sounddevice as sd
from engines import QwenTTS


def main():
    print("Initializing QwenTTS...")
    tts = QwenTTS()

    while True:
        try:
            print("\nEnter text to synthesize (or 'q' to quit):")
            text = input("> ")

            if text.lower() == 'q':
                break

            if not text:
                continue

            print("Synthesizing with Qwen3-TTS...")
            audio, sr = tts.synthesize(text)

            if audio is not None:
                print(f"Playing response... (Sampling rate: {sr}Hz)")
                sd.play(audio, sr)
                sd.wait()
            else:
                print("Failed to produce audio.")

        except KeyboardInterrupt:
            print("\nExiting TTS Test...")
            break
        except Exception as e:
            print(f"Error during TTS test: {e}")
            break


if __name__ == "__main__":
    main()
