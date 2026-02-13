import numpy as np
from qwen_tts import Qwen3TTSModel


class QwenTTS:
    def __init__(self):
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            trust_remote_code=True,
            device_map="auto"
        )

    def synthesize(
        self, text: str, speaker: str = "ono_anna", language: str = "japanese"
    ) -> tuple[np.ndarray | None, int]:
        wavs, sr = self.model.generate_custom_voice(text, speaker=speaker, language=language)
        if wavs and len(wavs) > 0:
            return wavs[0], sr
        return None, 16000
