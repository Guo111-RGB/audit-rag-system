# check_index_content.py
# 作用：查看索引中真实存储的前5条法律条款，检查是否只有编号没有内容

import pickle
import os

# 读取 BM25 索引文件
bm25_path = "bm25_clause.pkl"

if not os.path.exists(bm25_path):
    print(f"❌ 错误：找不到 {bm25_path}，请先运行构建脚本")
    exit(1)

with open(bm25_path, "rb") as f:
    _, chunks = pickle.load(f)

print(f"✅ 索引中共有 {len(chunks)} 个条款块\n")
print("=" * 60)
print("前 5 条条款内容预览（只显示前 300 个字符）：")
print("=" * 60)

for i in range(min(5, len(chunks))):
    print(f"\n--- 条款 {i+1} ---")
    content = chunks[i].page_content
    print(f"内容长度：{len(content)} 个字符")
    print(f"内容预览：{content[:300]}")
    if len(content) < 50:
        print("⚠️ 警告：这条内容很短，可能只有编号，没有正文！")
    print("-" * 40)

# 额外检查：专门搜索个人信息相关条款
print("\n" + "=" * 60)
print("🔍 搜索包含'个人信息'的条款：")
print("=" * 60)

found_personal = False
for i, chunk in enumerate(chunks):
    if "个人信息" in chunk.page_content:
        print(f"\n✅ 找到包含'个人信息'的条款（索引 {i}）：")
        print(chunk.page_content[:500])
        found_personal = True
        break

if not found_personal:
    print("❌ 未找到任何包含'个人信息'的条款！")
    print("   这说明你的索引确实丢失了具体法条内容，需要修复分割脚本。")