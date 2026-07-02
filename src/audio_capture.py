import numpy as np
import sounddevice as sd
from typing import Optional
import threading
import queue


class AudioCapture:
    """实时音频采集"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False

    def start_stream(self, callback):
        """启动音频流"""
        self.is_recording = True

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"音频状态: {status}")
            if self.is_recording:
                # 将音频数据放入队列
                self.audio_queue.put(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=audio_callback,
            dtype=np.float32
        )
        self.stream.start()
        print("🎤 音频流已启动...")

    def stop_stream(self):
        """停止音频流"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("🎤 音频流已停止")

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """获取音频块"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
