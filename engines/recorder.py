import sounddevice as sd
import numpy as np


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._audio_data = []
        self._stream = None

    def start(self):
        self._audio_data = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._callback
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._audio_data:
            return np.array([], dtype='float32')
        return np.concatenate(self._audio_data).flatten()

    def _callback(self, indata, frames, time, status):
        self._audio_data.append(indata.copy())
