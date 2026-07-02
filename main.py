#!/usr/bin/env python3
"""
智能空调精灵助手 - 命令行交互版
"""

import sys
import argparse
from src.assistant import ACGeniusAssistant


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════╗
║                                                   ║
║     🌬️  智能空调精灵助手 AC Genius Assistant     ║
║                                                   ║
║     基于 SenseVoice + 向量检索 + 热管理知识       ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
    """
    print(banner)


def interactive_mode(assistant: ACGeniusAssistant):
    """交互模式"""
    print("\n💡 提示: 输入问题，输入 'quit' 或 'exit' 退出\n")

    while True:
        try:
            query = input("🤔 您的问题: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！祝您清凉一夏~")
                break

            if not query:
                continue

            # 处理查询
            result = assistant.process_text_query(query)

            if result["success"]:
                print(f"\n✨ 【{result['category']}】")
                print(f"📢 {result['answer']}")
                print(f"🎯 置信度: {result['confidence']:.2%}")
                print(f"⏱️  耗时: {result['time_cost']:.3f}s")

                if result['related_questions']:
                    print("\n💭 相关问题:")
                    for i, q in enumerate(result['related_questions'], 1):
                        print(f"   {i}. {q}")
            else:
                print(f"\n❌ {result['error']}")

            print("\n" + "─" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="智能空调精灵助手")
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="运行模式"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="单次查询"
    )

    args = parser.parse_args()

    print_banner()

    # 初始化助手
    assistant = ACGeniusAssistant()

    if args.query:
        # 单次查询
        result = assistant.process_text_query(args.query)
        if result["success"]:
            print(f"\n问题: {result['query']}")
            print(f"答案: {result['answer']}")
        else:
            print(f"错误: {result['error']}")
    else:
        # 交互模式
        interactive_mode(assistant)


if __name__ == "__main__":
    main()
