# test_client.py - 多轮对话 + 可切换模式
import requests
import json

BASE_URL = "http://127.0.0.1:8000"


def search_mode():
    """纯检索模式：连续查询关键词，显示原文"""
    print("\n🔍 纯检索模式（只返回原文，不调用大模型）")
    print("输入关键词查询，输入 back 返回主菜单，输入 exit 退出\n")
    while True:
        query = input("检索词: ").strip()
        if query.lower() == "back":
            break
        if query.lower() in ["exit", "quit"]:
            exit(0)
        if not query:
            continue

        try:
            resp = requests.post(f"{BASE_URL}/search", json={"query": query, "top_k": 3})
            if resp.status_code == 200:
                data = resp.json()
                print("\n📄 检索结果：")
                for idx, item in enumerate(data["results"], 1):
                    print(f"\n--- 结果 {idx} ---")
                    print(item["content"][:500])
                    print(f"来源: {item['metadata']}")
            else:
                print(f"错误: {resp.text}")
        except Exception as e:
            print(f"连接失败: {e}")
        print()


def chat_mode():
    """对话模式：调用大模型生成答案，连续多轮"""
    print("\n💬 对话模式（通义千问生成）")
    print("输入问题，输入 back 返回主菜单，输入 exit 退出\n")
    while True:
        question = input("你: ").strip()
        if question.lower() == "back":
            break
        if question.lower() in ["exit", "quit"]:
            exit(0)
        if not question:
            continue

        payload = {"question": question, "use_llm": True}
        try:
            resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                print("\n🤖 回答：")
                print(data["answer"])
                print("\n📚 参考来源：")
                for ref in data["references"]:
                    print(f"  - {ref}")
            else:
                print(f"错误: {resp.text}")
        except Exception as e:
            print(f"连接失败: {e}")
        print()


def main():
    while True:
        print("\n=== 智能审计助手 ===")
        print("1. 纯检索模式（关键词→原文）")
        print("2. 对话模式（大模型生成）")
        print("0. 退出")
        choice = input("请选择: ").strip()

        if choice == "1":
            search_mode()
        elif choice == "2":
            chat_mode()
        elif choice == "0":
            print("再见！")
            break
        else:
            print("无效输入，请输入 1、2 或 0")


if __name__ == "__main__":
    main()