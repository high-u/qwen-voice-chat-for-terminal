import sounddevice as sd
import numpy as np
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
    print("Initializing QwenASR...")
    asr = QwenASR()

    while True:
        try:
            audio = record_audio()

            if audio.size == 0:
                print("No audio recorded.")
                continue

            print("Transcribing with Qwen3-ASR...")
            text = asr.transcribe(audio)

            print("\n--- ASR Result ---")
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
