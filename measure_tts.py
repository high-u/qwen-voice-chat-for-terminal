import pickle
import time
import argparse
from pathlib import Path

import sounddevice as sd
import torch
from qwen_tts import Qwen3TTSModel

PRESETS_DIR = Path(__file__).parent / "voice_presets"


def main():
    parser = argparse.ArgumentParser(description="Measure TTS synthesis time and play")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--voice", "-v", default="my_voice", help="Voice preset name")
    parser.add_argument("--mode", choices=["gpu", "cpu-offload"], default="gpu",
                        help="Execution mode: gpu (default) or cpu-offload")
    args = parser.parse_args()

    device_map = "cuda" if args.mode == "gpu" else "cpu"

    # モデル読み込み
    print(f"Loading model (mode: {args.mode}, voice: {args.voice})...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        trust_remote_code=True,
        device_map=device_map,
        dtype=torch.bfloat16,
    )

    # プリセット読み込み
    preset_path = PRESETS_DIR / f"{args.voice}.pkl"
    with open(preset_path, "rb") as f:
        preset_data = pickle.load(f)
    voice_clone_prompt = preset_data["prompt"]

    # 変換
    print(f"Synthesizing: {args.text}")
    start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language="Japanese",
        voice_clone_prompt=voice_clone_prompt,
    )
    elapsed = time.time() - start
    print(f"Synthesis time: {elapsed:.3f}s")

    # 再生
    if wavs and len(wavs) > 0:
        print(f"Playing... (sr: {sr}Hz, duration: {len(wavs[0]) / sr:.3f}s)")
        sd.play(wavs[0], sr)
        sd.wait()
    else:
        print("Failed to produce audio.")


if __name__ == "__main__":
    main()
