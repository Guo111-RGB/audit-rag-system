import pickle

with open("bm25_corpus.pkl", "rb") as f:
    _, chunks = pickle.load(f)

found = False
for i, chunk in enumerate(chunks):
    if "第一千零三十四条" in chunk.page_content:
        print(f"✅ 找到块 {i} 包含第一千零三十四条")
        print("内容预览：")
        print(chunk.page_content[:500])
        found = True
        break

if not found:
    print("❌ 未找到任何包含'第一千零三十四条'的块")
    # 打印包含“第一千零三十”的所有块，看看最近的条款编号
    print("\n包含'第一千零三十'的块：")
    for i, chunk in enumerate(chunks):
        if "第一千零三十" in chunk.page_content:
            print(f"块 {i}: {chunk.page_content[:100]}...")