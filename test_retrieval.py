# test_retrieval.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.hybrid_retriever import HybridRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chroma_dir = os.path.join(BASE_DIR, "chroma_db_clause")
bm25_path = os.path.join(BASE_DIR, "bm25_clause.pkl")

retriever = HybridRetriever(chroma_dir, bm25_path)

# 测试合同条款
test_query = "收集用户姓名、身份证号、手机号、银行卡号用于营销推广"

print("="*60)
print("测试检索器是否正常工作")
print("="*60)

results = retriever.hybrid_search(test_query, k=5, alpha=0.7)

if results:
    print(f"\n✅ 成功检索到 {len(results)} 条相关法条：")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- 结果 {i} (得分: {score:.4f}) ---")
        print(f"来源: {doc.metadata.get('source', '未知')}")
        print(f"条款: {doc.metadata.get('clause_id', '未知')}")
        print(f"内容: {doc.page_content[:200]}...")
else:
    print("\n❌ 检索结果为空！请检查索引是否正常。")