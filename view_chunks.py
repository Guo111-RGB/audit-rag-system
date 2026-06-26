# view_chunks.py
import pickle

# 加载 BM25 保存的 chunks（如果还未构建 BM25，也可以直接从向量库读取，但 BM25 文件包含完整 chunks）
with open("bm25_corpus.pkl", "rb") as f:
    tokenized_corpus, chunks = pickle.load(f)

print(f"共有 {len(chunks)} 个文本块\n")
print("="*60)

# 查看前 5 个块的内容（每个块显示前 300 字符）
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- 块 {i+1} ---")
    print(f"来源: {chunk.metadata.get('source', '未知')}")
    print(f"内容预览: {chunk.page_content[:300]}...")
    print("-"*60)

# 可以选择输入块索引查看完整内容
while True:
    try:
        idx = input("\n输入块索引查看完整内容（1-{}），输入 q 退出: ".format(len(chunks)))
        if idx.lower() == 'q':
            break
        idx = int(idx) - 1
        if 0 <= idx < len(chunks):
            print(f"\n--- 块 {idx+1} 完整内容 ---")
            print(chunks[idx].page_content)
        else:
            print("索引超出范围")
    except ValueError:
        print("请输入数字")