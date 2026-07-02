from funasr import AutoModel
import numpy as np
from typing import Optional


class SenseVoiceASR:
    """SenseVoice 语音识别引擎"""

    def __init__(self, model_name: str = "iic/SenseVoiceSmall"):
        print("🔄 正在加载 SenseVoice 模型...")
        self.model = AutoModel(
            model=model_name,
            device="cpu",  # Jetson 上可改为 "cuda"
            disable_update=True
        )
        print("✅ SenseVoice 模型加载完成")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """语音转文字"""
        try:
            # SenseVoice 输入格式
            result = self.model.generate(
                input=audio,
                batch_size=1,
                language="auto",  # 自动检测语言
            )

            if result and len(result) > 0:
                text = result[0]["text"]
                return text.strip()
            return None

        except Exception as e:
            print(f"❌ 识别错误: {e}")
            return None

    def transcribe_file(self, audio_path: str) -> Optional[str]:
        """识别音频文件"""
        try:
            result = self.model.generate(
                input=audio_path,
                batch_size=1,
                language="auto",
            )

            if result and len(result) > 0:
                return result[0]["text"].strip()
            return None

        except Exception as e:
            print(f"❌ 文件识别错误: {e}")
            return None
