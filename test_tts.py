import time
import argparse
from engines import QwenTTS


def main():
    parser = argparse.ArgumentParser(description="Test QwenTTS with GPU or CPU-offload")
    parser.add_argument("--mode", choices=["gpu", "cpu-offload"], default="gpu",
                        help="Execution mode: gpu (default) or cpu-offload")
    parser.add_argument("--voice", default="my_voice",
                        help="Voice preset name (default: my_voice)")
    args = parser.parse_args()

    # モードに応じて設定
    if args.mode == "gpu":
        device_map = "cuda"
    else:  # cpu
        device_map = "cpu"

    print(f"Initializing QwenTTS (mode: {args.mode}, voice: {args.voice})...")
    tts = QwenTTS(voice_name=args.voice, device_map=device_map)

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
            success, sr = tts.synthesize(text)
            elapsed = time.time() - start
            print(f"Synthesis time: {elapsed:.3f}s")

            if not success:
                print("Failed to produce audio.")

        except KeyboardInterrupt:
            print("\nExiting TTS Test...")
            break
        except Exception as e:
            print(f"Error during TTS test: {e}")
            break


if __name__ == "__main__":
    main()
