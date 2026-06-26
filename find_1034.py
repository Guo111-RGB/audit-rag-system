# find_1034.py
import pickle

with open("bm25_corpus.pkl", "rb") as f:
    _, chunks = pickle.load(f)

found = False
for i, chunk in enumerate(chunks):
    if "第一千零三十四条" in chunk.page_content or "1034" in chunk.page_content:
        print(f"✅ 找到块 {i} 包含目标条款")
        print("内容前500字符：")
        print(chunk.page_content[:500])
        found = True
        break

if not found:
    print("❌ 未找到任何包含 '第一千零三十四条' 或 '1034' 的块")
    # 检查是否包含“个人信息”字样，看人格权编是否存在
    for i, chunk in enumerate(chunks):
        if "个人信息" in chunk.page_content:
            print(f"\n包含'个人信息'的块 {i}: {chunk.page_content[:200]}...")
            break
    else:
        print("未找到任何包含'个人信息'的块，人格权编可能完全缺失")