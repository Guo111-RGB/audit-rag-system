# debug_retrieval.py
import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.hybrid_retriever import HybridRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chroma_dir = os.path.join(BASE_DIR, "chroma_db_full")
bm25_path = os.path.join(BASE_DIR, "bm25_corpus.pkl")

retriever = HybridRetriever(chroma_dir, bm25_path)

query = "民法典第一千零三十四条"
results = retriever.hybrid_search(query, k=5, alpha=0.3)

print(f"\n查询：{query}\n")
for i, (doc, score) in enumerate(results):
    print(f"--- 结果 {i+1} (得分: {score:.4f}) ---")
    # 只显示前200个字符以便观察
    print(doc.page_content[:400])
    print(f"来源页: {doc.metadata.get('page_label', '?')}\n")