#!/usr/bin/env python3
"""
Voice Clone Tool using Qwen3-TTS Base model.

This tool allows you to:
1. Record audio and create a reusable voice clone prompt
2. Synthesize and play speech using a saved voice clone prompt
3. List and manage saved voice clone prompts

Usage:
    # Create a voice clone prompt (records from microphone)
    python voice_clone.py create --name my_voice --text "Reference audio transcript"

    # Synthesize and play speech (single text)
    python voice_clone.py synthesize --name my_voice --text "Text to synthesize"

    # Synthesize and play multiple texts
    python voice_clone.py synthesize --name my_voice --text "First text" --text "Second text"

    # List saved prompts
    python voice_clone.py list

    # Delete a saved prompt
    python voice_clone.py delete --name my_voice
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from qwen_tts import Qwen3TTSModel


# Default directory for saving voice clone prompts
PROMPTS_DIR = Path(__file__).parent / "voice_presets"
SAMPLE_RATE = 16000


def record_audio(fs: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from microphone."""
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


class VoiceCloner:
    """Voice cloning using Qwen3-TTS Base model."""

    def __init__(self, device_map: str = "auto", model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
        print(f"Loading model: {model_name}")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            dtype=torch.bfloat16,
        )
        self.prompts_dir = PROMPTS_DIR
        self.prompts_dir.mkdir(exist_ok=True)

    def create_prompt_from_recording(
        self,
        name: str,
        ref_text: str,
        x_vector_only_mode: bool = False,
    ) -> Path:
        """
        Record audio and create a voice clone prompt.

        Args:
            name: Name for the saved prompt
            ref_text: Transcript of what will be recorded
            x_vector_only_mode: If True, only use speaker embedding (lower quality but ref_text not required)

        Returns:
            Path to the saved prompt file
        """
        print(f"\nCreating voice clone prompt '{name}'")
        print(f"Please say: \"{ref_text}\"")

        audio = record_audio(SAMPLE_RATE)

        if audio.size == 0:
            print("No audio recorded. Aborting.")
            return None

        print("Processing voice clone prompt...")
        prompt_data = self.model.create_voice_clone_prompt(
            ref_audio=(audio, SAMPLE_RATE),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        # Save metadata along with prompt (audio data is NOT saved, only the extracted features)
        save_data = {
            "prompt": prompt_data,
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode,
        }

        prompt_path = self.prompts_dir / f"{name}.pkl"
        with open(prompt_path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"Prompt saved to: {prompt_path}")
        return prompt_path

    def load_prompt(self, name: str) -> dict:
        """Load a saved voice clone prompt."""
        prompt_path = self.prompts_dir / f"{name}.pkl"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt '{name}' not found at {prompt_path}")

        with open(prompt_path, "rb") as f:
            return pickle.load(f)

    def synthesize_and_play(
        self,
        texts: list[str],
        prompt_name: str,
        language: str = "Japanese",
    ):
        """
        Synthesize multiple texts and play them sequentially.

        Args:
            texts: List of texts to synthesize
            prompt_name: Name of the saved prompt
            language: Language for synthesis
        """
        print(f"Loading prompt: {prompt_name}")
        saved_data = self.load_prompt(prompt_name)
        prompt = saved_data["prompt"]

        print(f"Synthesizing {len(texts)} text(s)...")
        wavs, sr = self.model.generate_voice_clone(
            text=texts,
            language=[language] * len(texts),
            voice_clone_prompt=prompt,
        )

        if wavs:
            for i, audio in enumerate(wavs):
                preview = texts[i][:30] + "..." if len(texts[i]) > 30 else texts[i]
                print(f"[{i+1}/{len(texts)}] Playing: {preview}")
                sd.play(audio, sr)
                sd.wait()
        else:
            print("Failed to synthesize audio.")

    def list_prompts(self) -> list[dict]:
        """List all saved voice clone prompts."""
        prompts = []
        for pkl_file in self.prompts_dir.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                prompts.append({
                    "name": pkl_file.stem,
                    "ref_text": data.get("ref_text", "N/A")[:50] + "..." if len(data.get("ref_text", "")) > 50 else data.get("ref_text", "N/A"),
                    "x_vector_only_mode": data.get("x_vector_only_mode", False),
                })
            except Exception as e:
                prompts.append({
                    "name": pkl_file.stem,
                    "error": str(e),
                })
        return prompts

    def delete_prompt(self, name: str) -> bool:
        """Delete a saved voice clone prompt."""
        prompt_path = self.prompts_dir / f"{name}.pkl"
        if prompt_path.exists():
            prompt_path.unlink()
            print(f"Deleted prompt: {name}")
            return True
        print(f"Prompt '{name}' not found")
        return False


def cmd_create(args):
    """Handle 'create' command."""
    cloner = VoiceCloner(device_map=args.device)
    cloner.create_prompt_from_recording(
        name=args.name,
        ref_text=args.text,
        x_vector_only_mode=args.x_vector_only,
    )


def cmd_synthesize(args):
    """Handle 'synthesize' command."""
    cloner = VoiceCloner(device_map=args.device)
    cloner.synthesize_and_play(
        texts=args.text,
        prompt_name=args.name,
        language=args.language,
    )


def cmd_list(args):
    """Handle 'list' command."""
    # Don't load model for listing
    prompts_dir = PROMPTS_DIR
    prompts = []
    for pkl_file in prompts_dir.glob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            prompts.append({
                "name": pkl_file.stem,
                "ref_text": data.get("ref_text", "N/A")[:50] + "..." if len(data.get("ref_text", "")) > 50 else data.get("ref_text", "N/A"),
                "x_vector_only_mode": data.get("x_vector_only_mode", False),
            })
        except Exception as e:
            prompts.append({
                "name": pkl_file.stem,
                "error": str(e),
            })

    if not prompts:
        print("No saved voice clone prompts found.")
        print(f"Prompts directory: {PROMPTS_DIR}")
        return

    print(f"Saved voice clone prompts ({len(prompts)}):")
    print("-" * 60)
    for p in prompts:
        if "error" in p:
            print(f"  {p['name']}: ERROR - {p['error']}")
        else:
            print(f"  {p['name']}:")
            print(f"    Ref text: {p['ref_text']}")
            print(f"    X-vector only: {p['x_vector_only_mode']}")
    print("-" * 60)
    print(f"Prompts directory: {PROMPTS_DIR}")


def cmd_delete(args):
    """Handle 'delete' command."""
    prompt_path = PROMPTS_DIR / f"{args.name}.pkl"
    if prompt_path.exists():
        prompt_path.unlink()
        print(f"Deleted prompt: {args.name}")
    else:
        print(f"Prompt '{args.name}' not found")


def main():
    parser = argparse.ArgumentParser(
        description="Voice Clone Tool using Qwen3-TTS Base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for model (default: auto)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Record audio and create a voice clone prompt")
    create_parser.add_argument("--name", required=True, help="Name for the saved prompt")
    create_parser.add_argument("--text", required=True, help="Transcript of what you will record")
    create_parser.add_argument("--x-vector-only", action="store_true",
                                help="Use x-vector only mode (lower quality, no transcript needed)")
    create_parser.set_defaults(func=cmd_create)

    # Synthesize command
    syn_parser = subparsers.add_parser("synthesize", help="Synthesize and play speech using a saved prompt")
    syn_parser.add_argument("--name", required=True, help="Name of the saved prompt")
    syn_parser.add_argument("--text", required=True, action="append",
                            help="Text to synthesize (can be specified multiple times)")
    syn_parser.add_argument("--language", default="Japanese", help="Language for synthesis")
    syn_parser.set_defaults(func=cmd_synthesize)

    # List command
    list_parser = subparsers.add_parser("list", help="List saved voice clone prompts")
    list_parser.set_defaults(func=cmd_list)

    # Delete command
    del_parser = subparsers.add_parser("delete", help="Delete a saved voice clone prompt")
    del_parser.add_argument("--name", required=True, help="Name of the prompt to delete")
    del_parser.set_defaults(func=cmd_delete)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
