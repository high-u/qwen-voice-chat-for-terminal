import pickle
import queue
import re
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from qwen_tts import Qwen3TTSModel


# Directory for voice clone presets
PRESETS_DIR = Path(__file__).parent.parent / "voice_presets"

# Sentence split pattern (keeps delimiters)
SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?])")


class QwenTTS:
    def __init__(self, voice_name: str, device_map: str = "auto"):
        if not voice_name:
            raise ValueError("voice_name is required. Create a preset with voice_clone.py first.")

        preset_path = PRESETS_DIR / f"{voice_name}.pkl"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Voice preset '{voice_name}' not found at {preset_path}. "
                f"Create it using: python voice_clone.py create --name {voice_name} --text \"...\""
            )

        print(f"Loading TTS model with voice preset: {voice_name}")
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            trust_remote_code=True,
            device_map=device_map,
            dtype=torch.bfloat16,
        )

        with open(preset_path, "rb") as f:
            preset_data = pickle.load(f)
        self.voice_clone_prompt = preset_data["prompt"]

    def _split_text(self, text: str) -> list[str]:
        """Split text by sentence delimiters, keeping delimiters."""
        sentences = SPLIT_PATTERN.split(text)
        # Filter empty strings and combine delimiter with sentence
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                if result and s in "。！？.!?":
                    result[-1] += s
                else:
                    result.append(s)
        return result

    def synthesize(
        self, text: str, language: str = "Japanese"
    ) -> tuple[bool, int]:
        """
        Synthesize and play speech with streaming (non-blocking playback).

        Returns:
            Tuple of (success, sample_rate).
            Audio playback is handled internally.
        """
        sentences = self._split_text(text)

        if not sentences:
            return False, 16000

        if len(sentences) == 1:
            # Single sentence: generate and play directly
            wavs, sr = self.model.generate_voice_clone(
                text=sentences[0],
                language=language,
                voice_clone_prompt=self.voice_clone_prompt,
            )
            if wavs and len(wavs) > 0:
                sd.play(wavs[0], sr)
                sd.wait()
                return True, sr
            return False, 16000

        # Multiple sentences: use streaming playback
        audio_queue = queue.Queue()
        sample_rate = [16000]
        has_error = [False]

        def producer():
            """Generate audio and put into queue."""
            for i, sentence in enumerate(sentences):
                try:
                    wavs, sr = self.model.generate_voice_clone(
                        text=sentence,
                        language=language,
                        voice_clone_prompt=self.voice_clone_prompt,
                    )
                    if wavs and len(wavs) > 0:
                        sample_rate[0] = sr
                        audio_queue.put(wavs[0])
                    else:
                        audio_queue.put(None)
                except Exception as e:
                    print(f"Error synthesizing sentence {i+1}: {e}")
                    has_error[0] = True
                    audio_queue.put(None)
            # Signal end of production
            audio_queue.put(None)

        def consumer():
            """Play audio from queue sequentially."""
            while True:
                audio = audio_queue.get()
                if audio is None:
                    break
                sd.play(audio, sample_rate[0])
                sd.wait()

        # Start threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        # Wait for completion
        producer_thread.join()
        consumer_thread.join()

        return not has_error[0], sample_rate[0]
