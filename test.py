#!/usr/bin/env python3
"""
语音交互测试脚本
"""

import numpy as np
import wave
from src.assistant import ACGeniusAssistant


def test_audio_file(assistant: ACGeniusAssistant, audio_path: str):
    """测试音频文件"""
    print(f"🎵 正在处理音频文件: {audio_path}")

    # 读取 wav 文件
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # 处理查询
    result = assistant.process_voice_query(audio_array, sample_rate)

    if result["success"]:
        print(f"\n✅ 识别成功!")
        print(f"📝 问题: {result['query']}")
        print(f"💡 答案: {result['answer']}")
        print(f"🏷️  分类: {result['category']}")
        print(f"🎯 置信度: {result['confidence']:.2%}")
    else:
        print(f"\n❌ 处理失败: {result.get('error', '未知错误')}")


def test_text_queries(assistant: ACGeniusAssistant):
    """测试文本查询"""
    test_cases = [
        "空调不制冷了怎么办",
        "空调一直开着费电吗",
        "夏天空调开26度够吗",
        "空调滴水是坏了吗",
        "变频空调值得买吗"
    ]

    print("\n" + "="*60)
    print("📋 开始批量测试...")
    print("="*60 + "\n")

    for i, query in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] 问题: {query}")
        result = assistant.process_text_query(query)

        if result["success"]:
            print(f"✅ 答案: {result['answer'][:100]}...")
            print(f"   分类: {result['category']} | 置信度: {result['confidence']:.2%}")
        else:
            print(f"❌ 错误: {result['error']}")

        print()


if __name__ == "__main__":
    print("🧪 测试模式启动\n")

    # 初始化助手
    assistant = ACGeniusAssistant()

    # 文本测试
    test_text_queries(assistant)

    print("\n✅ 测试完成!")
