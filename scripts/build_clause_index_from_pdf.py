# scripts/build_clause_index_from_pdf.py
#文件遍历：使用 glob.glob(os.path.join(PDF_DIR, "*.pdf")) 获取所有 PDF 文件路径。
# 逐个处理：对每个 PDF 提取文本、分割条款，并将生成的 Document 追加到 all_documents 列表中。
# 元数据区分：在 metadata 中添加 "source": filename，方便后续知道条款来自哪个文档。
# 统一构建：所有 PDF 处理完后，统一构建向量库和 BM25 索引（覆盖旧索引）。
# 错误处理：如果某个 PDF 提取失败，会跳过并继续处理其他文件。
import os
import re
import pickle
import glob
import pdfplumber
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
import jieba
import shutil

# 配置
PDF_DIR = "data/full_docs/"                    # PDF 存放目录
CHROMA_DIR = "chroma_db_clause"                # 向量库输出目录
BM25_PATH = "bm25_clause.pkl"                  # BM25 索引输出文件

def extract_text_from_pdf(pdf_path):
    """用 pdfplumber 提取单个 PDF 的全部文字"""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def split_into_clauses(text):
    """按「第X条」或「第一千零X条」分割文本"""
    pattern = r'(第[一二三四五六七八九十百千万零]+条)'
    parts = re.split(pattern, text)
    clauses = []
    for i in range(1, len(parts), 2):
        number = parts[i].strip()
        content = (parts[i+1] if i+1 < len(parts) else "").strip()
        if content:
            clauses.append((number, content))
    return clauses

def main():
    # 处理 data/full_docs/ 下所有 PDF
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print("❌ 错误：未找到任何 PDF 文件")
        return
    print(f"✅ 找到 {len(pdf_files)} 个 PDF 文件")

    all_documents = []  # 存储所有条款的 Document 对象
    total_clauses = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"正在处理: {filename}")
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            if not raw_text.strip():
                print(f"  警告：{filename} 提取文字为空，跳过")
                continue
            clause_list = split_into_clauses(raw_text)
            print(f"  提取到 {len(clause_list)} 个条款")
            for num, content in clause_list:
                # 添加 source 元数据以区分来源
                doc = Document(
                    page_content=f"{num} {content}",
                    metadata={"clause_id": num, "source": filename}
                )
                all_documents.append(doc)
            total_clauses += len(clause_list)
        except Exception as e:
            print(f"  处理 {filename} 时出错: {e}")

    if not all_documents:
        print("错误：未提取到任何有效条款，请检查 PDF 文件或分割逻辑")
        return

    print(f"总计提取 {total_clauses} 个条款，来自 {len(pdf_files)} 个文档")

    # 构建向量库
    print("正在构建向量库...")
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    # 如果已存在则删除旧库
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    vector_store = Chroma.from_documents(
        all_documents, embeddings, persist_directory=CHROMA_DIR
    )
    print(f"向量库已保存到 {CHROMA_DIR}")

    # 构建 BM25 索引
    print("正在构建 BM25 索引...")
    tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in all_documents]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump((tokenized_corpus, all_documents), f)
    print(f"BM25 索引已保存到 {BM25_PATH}")

    print("索引构建完成！")

if __name__ == "__main__":
    main()