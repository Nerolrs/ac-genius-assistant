import os
from typing import Optional, List, Dict
from src.vector_store import VectorStore
from src.thermal_knowledge import THERMAL_KNOWLEDGE_BASE
import time

try:
    from src.asr_engine import SenseVoiceASR
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    print("[WARN] ASR模块不可用，语音功能将被禁用")


class ACGeniusAssistant:
    """智能空调精灵助手核心引擎"""

    def __init__(self):
        print("🌟 初始化空调精灵助手...")

        # 初始化语音识别（可选）
        if ASR_AVAILABLE:
            self.asr = SenseVoiceASR()
        else:
            self.asr = None

        # 初始化向量库
        self.vector_store = VectorStore()

        # 加载知识库
        self._initialize_knowledge_base()

        print("✅ 空调精灵助手已就绪！")

    def _initialize_knowledge_base(self):
        """初始化知识库"""
        print("📚 正在加载热管理知识库...")
        self.vector_store.add_documents(THERMAL_KNOWLEDGE_BASE)

    def process_voice_query(self, audio_data, sample_rate: int = 16000) -> Dict:
        """处理语音查询"""
        if not self.asr:
            return {
                "success": False,
                "error": "语音识别模块未安装，请安装 funasr 库",
                "time_cost": 0
            }

        start_time = time.time()

        # 1. 语音识别
        print("🎤 正在识别语音...")
        text = self.asr.transcribe(audio_data, sample_rate)

        if not text:
            return {
                "success": False,
                "error": "语音识别失败",
                "time_cost": time.time() - start_time
            }

        print(f"📝 识别结果: {text}")

        # 2. 向量检索
        return self.process_text_query(text, time_cost_offset=time.time() - start_time)

    def process_text_query(self, query: str, time_cost_offset: float = 0) -> Dict:
        """处理文本查询"""
        start_time = time.time()

        print(f"🔍 正在检索: {query}")
        results = self.vector_store.search(query, top_k=3)

        if not results:
            return {
                "success": False,
                "query": query,
                "error": "未找到相关答案",
                "time_cost": time.time() - start_time + time_cost_offset
            }

        # 取最相关的答案
        best_match = results[0]

        response = {
            "success": True,
            "query": query,
            "answer": best_match["answer"],
            "category": best_match["category"],
            "confidence": best_match["score"],
            "related_questions": [r["question"] for r in results[1:3]],
            "time_cost": time.time() - start_time + time_cost_offset
        }

        return response

    def batch_query(self, queries: List[str]) -> List[Dict]:
        """批量查询"""
        results = []
        for query in queries:
            result = self.process_text_query(query)
            results.append(result)
        return results
