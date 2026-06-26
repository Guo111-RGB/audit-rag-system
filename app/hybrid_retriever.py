# app/hybrid_retriever.py
import os
import pickle
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import jieba
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class HybridRetriever:
    """
    混合检索器：同时使用向量检索（语义）和 BM25（关键词）
    """

    def __init__(self, chroma_dir: str, bm25_pkl_path: str):
        print("初始化混合检索器...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self.embeddings
        )
        with open(bm25_pkl_path, "rb") as f:
            self.tokenized_corpus, self.chunks = pickle.load(f)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 加载完成，共 {len(self.chunks)} 个文档块")

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max 归一化到 [0,1]"""
        min_s = np.min(scores)
        max_s = np.max(scores)
        if max_s - min_s < 1e-8:
            return np.ones_like(scores) * 0.5
        return (scores - min_s) / (max_s - min_s)

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Tuple[Document, float]]:
        """
        混合检索
        alpha: 向量权重，BM25 权重为 1-alpha
        """
        # ---------- 1. 向量检索 ----------
        # 多取一些结果用于融合（扩大候选池）
        # ========== 新增：数字转换预处理 ==========
        import re
        import cn2an  # 确保文件顶部已导入 cn2an

        # 1. 处理 "第数字条" 格式（如 "第1034条"）
        def convert_clause(match):
            num_str = match.group(1)
            try:
                chinese_num = cn2an.an2cn(num_str, mode='low')
                return f"第{chinese_num}条"
            except:
                return match.group(0)

        query = re.sub(r'第(\d+)条', convert_clause, query)

        # 2. 处理 "数字条" 格式（如 "1034条"），前面没有"第"
        def convert_plain(match):
            num_str = match.group(1)
            try:
                chinese_num = cn2an.an2cn(num_str, mode='low')
                return f"第{chinese_num}条"
            except:
                return match.group(0)

        query = re.sub(r'(?<!第)(\d+)条', convert_plain, query)
        # ========== 转换结束 ==========

        vec_results = self.vector_store.similarity_search_with_score(query, k=k * 2)
        vec_score_map = {}
        for doc, dist in vec_results:
            # Chroma 返回的是 L2 距离，范围大约 0-2，转换成相似度（越大越好）
            sim = 1.0 - (dist / 2.0)
            vec_score_map[doc.page_content] = sim

        # ---------- 2. BM25 检索 ----------
        tokenized_query = list(jieba.cut(query))
        bm25_raw = self.bm25.get_scores(tokenized_query)  # 长度 = 总文档块数
        bm25_norm = self._normalize(bm25_raw)

        # ---------- 3. 融合分数 ----------
        merged = []
        for idx, chunk in enumerate(self.chunks):
            vec_score = vec_score_map.get(chunk.page_content, 0.0)
            combined = alpha * vec_score + (1 - alpha) * bm25_norm[idx]
            merged.append((chunk, combined))

        # 按综合分数降序排序
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]


    def search(self, query: str, k: int = 5) -> List[Document]:
        """简化版：只返回文档列表"""
        return [doc for doc, _ in self.hybrid_search(query, k)]

    def arabic_to_chinese(num):
        """
        将阿拉伯数字（0-9999）转换为中文数字字符串（不含"第"和"条"）
        示例：1034 -> "一千零三十四"
        """
        if num == 0:
            return "零"
        digits = "零一二三四五六七八九"
        units = ["", "十", "百", "千"]
        result = ""
        # 处理万位及以上（简化，仅支持到9999）
        if num >= 1000:
            result += digits[num // 1000] + "千"
            num %= 1000
        if num >= 100:
            if result and num // 100 == 0:
                result += "零"
            else:
                result += digits[num // 100] + "百"
            num %= 100
        if num >= 10:
            if result and num // 10 == 0:
                result += "零"
            else:
                if num // 10 == 1 and not result:
                    result += "十"
                else:
                    result += digits[num // 10] + "十"
            num %= 10
        if num > 0:
            if result and not result.endswith("零"):
                result += "零"
            result += digits[num]
        # 处理特殊情况：十位为1且没有百位以上时，如"一十"应显示为"十"
        if result.startswith("一十"):
            result = result[1:]
        return result