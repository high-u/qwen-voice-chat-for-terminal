import sounddevice as sd
import numpy as np
import time
import argparse
from engines import QwenASR


def record_audio(fs=16000):
    print("\nPress Enter to start recording...")
    input()

    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    print("Recording... Press Enter to stop.")
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', callback=callback):
        input()

    print("Recording finished.")
    if not audio_data:
        return np.array([], dtype='float32')

    return np.concatenate(audio_data).flatten()


def main():
    parser = argparse.ArgumentParser(description="Test QwenASR with GPU or CPU-offload")
    parser.add_argument("--mode", choices=["gpu", "cpu-offload"], default="gpu",
                        help="Execution mode: gpu (default) or cpu-offload")
    args = parser.parse_args()

    # モードに応じて設定
    if args.mode == "gpu":
        device_map = "cuda"
    else:  # cpu
        device_map = "cpu"

    print(f"Initializing QwenASR (mode: {args.mode})...")
    asr = QwenASR(device_map=device_map)

    while True:
        try:
            audio = record_audio()

            if audio.size == 0:
                print("No audio recorded.")
                continue

            print("Transcribing with Qwen3-ASR...")
            start = time.time()
            text = asr.transcribe(audio)
            elapsed = time.time() - start

            print(f"\n--- ASR Result (time: {elapsed:.3f}s) ---")
            print(text)
            print("------------------\n")

            print("Press Enter to test again, Ctrl+C to exit.")
            input()

        except KeyboardInterrupt:
            print("\nExiting ASR Test...")
            break
        except Exception as e:
            print(f"Error during ASR test: {e}")
            break


if __name__ == "__main__":
    main()
