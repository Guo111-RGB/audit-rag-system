# scripts/build_fulltext_index.py
"""
专门为民法典和金融法构建索引
分块大小: chunk_size=1000, overlap=200
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import pickle
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
import jieba

# ================== 配置 ==================
FULL_DOCS_DIR = os.path.join(BASE_DIR, "data", "full_docs")
CHROMA_FULL_DIR = os.path.join(BASE_DIR, "chroma_db_full")
BM25_PKL_PATH = os.path.join(BASE_DIR, "bm25_corpus.pkl")

# 指定要处理的文件列表（只处理这两个PDF）
TARGET_FILES = ["min_fa_dian.pdf", "financial_law.pdf"]


# ================== 函数定义 ==================
def load_target_documents():
    """只加载指定的两个 PDF 文件"""
    docs = []
    for filename in TARGET_FILES:
        filepath = os.path.join(FULL_DOCS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"警告：文件不存在 - {filepath}，跳过")
            continue
        try:
            loader = PyPDFLoader(filepath)
            print(f"正在加载 PDF: {filename}")
            pages = loader.load()
            docs.extend(pages)
            print(f"  成功加载 {len(pages)} 页")
        except Exception as e:
            print(f"加载失败 {filename}: {e}")
    return docs


def chunk_documents(docs):
    """分块：chunk_size=1000, overlap=200"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"分块完成，共 {len(chunks)} 个文本块")
    return chunks


def build_vector_store(chunks):
    """构建 Chroma 向量库"""
    print("正在加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    if os.path.exists(CHROMA_FULL_DIR):
        shutil.rmtree(CHROMA_FULL_DIR)
        print("已删除旧向量库")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FULL_DIR
    )
    print(f"向量库已保存到 {CHROMA_FULL_DIR}")
    return vector_store


def build_bm25_index(chunks):
    """构建 BM25 索引"""
    print("正在构建 BM25 索引，使用 jieba 分词...")
    tokenized_corpus = []
    for idx, chunk in enumerate(chunks):
        words = list(jieba.cut(chunk.page_content))
        tokenized_corpus.append(words)
        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1}/{len(chunks)} 个块")
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump((tokenized_corpus, chunks), f)
    print(f"BM25 索引已保存到 {BM25_PKL_PATH}")


def main():
    print("=" * 50)
    print("专门为民法典和金融法构建索引")
    print(f"目标文件: {', '.join(TARGET_FILES)}")
    print("分块参数: chunk_size=1000, overlap=200")
    print("=" * 50)

    docs = load_target_documents()
    if not docs:
        print("错误：未找到任何目标文档！请确保以下文件存在于 data/full_docs/ 目录：")
        for f in TARGET_FILES:
            print(f"  - {f}")
        return

    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    build_bm25_index(chunks)

    print("=" * 50)
    print("索引构建完成！")
    print(f"向量库: {CHROMA_FULL_DIR}")
    print(f"BM25: {BM25_PKL_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()