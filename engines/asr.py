import numpy as np
from qwen_asr import Qwen3ASRModel


class QwenASR:
    def __init__(self):
        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            trust_remote_code=True,
            device_map="auto"
        )

    def transcribe(self, audio: np.ndarray, sr: int = 16000, language: str = "Japanese") -> str:
        results = self.model.transcribe((audio, sr), language=language)
        if results and len(results) > 0:
            return results[0].text
        return ""
