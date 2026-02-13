import sounddevice as sd
import time
import argparse
from engines import QwenTTS


def main():
    parser = argparse.ArgumentParser(description="Test QwenTTS with GPU or CPU-offload")
    parser.add_argument("--mode", choices=["gpu", "cpu-offload"], default="gpu",
                        help="Execution mode: gpu (default) or cpu-offload")
    args = parser.parse_args()

    # モードに応じて設定
    if args.mode == "gpu":
        device_map = "cuda"
    else:  # cpu
        device_map = "cpu"

    print(f"Initializing QwenTTS (mode: {args.mode})...")
    tts = QwenTTS(device_map=device_map)

    while True:
        try:
            print("\nEnter text to synthesize (or 'q' to quit):")
            text = input("> ")

            if text.lower() == 'q':
                break

            if not text:
                continue

            print("Synthesizing with Qwen3-TTS...")
            start = time.time()
            audio, sr = tts.synthesize(text)
            elapsed = time.time() - start
            print(f"Synthesis time: {elapsed:.3f}s")

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
