# app/rag_retriever.py
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


#rag_retriever.py 只做检索（Search），不做生成（Generate）


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


class AuditRAG:
    def __init__(self):
        print("正在加载嵌入模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("正在连接向量数据库...")
        self.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def search(self, query: str):
        """根据问题检索最相关的审计条款"""
        docs = self.retriever.invoke(query)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": None  # Chroma默认不返回分数，如需可改用similarity_search_with_score
            })
        return results

    def search_with_score(self, query: str):
        """返回带相似度分数的检索结果"""
        docs_with_score = self.vector_store.similarity_search_with_score(query, k=5)
        results = []
        for doc, score in docs_with_score:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        return results


# 简单测试（可单独运行）
if __name__ == "__main__":
    rag = AuditRAG()
    query = "风险控制矩阵（RCM）"
    results = rag.search(query)
    for i, r in enumerate(results):
        print(f"\n--- 结果 {i + 1} ---")
        print(r["content"][:200])