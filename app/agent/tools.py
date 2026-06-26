# app/agent/tools.py
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from app.hybrid_retriever import HybridRetriever
import os

# 初始化检索器（复用你已有的）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
chroma_dir = os.path.join(BASE_DIR, "chroma_db_clause")
bm25_path = os.path.join(BASE_DIR, "bm25_clause.pkl")
retriever = HybridRetriever(chroma_dir, bm25_path)

# 引用全局的 llm（稍后在 main.py 中设置）
_llm = None


def set_llm(llm):
    global _llm
    _llm = llm


@tool
def search_law(query: str) -> str:
    """
    在法律法规知识库中检索相关条款。
    输入：法律问题或关键词（如"个人信息保护"、"逾期利率"）
    返回：相关法条原文及来源，最多返回前3条。
    """
    results = retriever.hybrid_search(query, k=5, alpha=0.7)
    if not results:
        return "未找到相关法律条款。"

    output = []
    for doc, score in results[:3]:
        source = doc.metadata.get("source", "未知")
        clause = doc.metadata.get("clause_id", "未知条款")
        content = doc.page_content[:500]  # 限制长度
        output.append(f"【来源：{source}】条款：{clause}\n{content}\n")
    return "\n".join(output)


@tool
def extract_clauses(contract_text: str) -> str:
    """
    从合同中提取关键条款（如违约责任、保密义务、争议解决等）。
    输入：完整的合同文本（可能很长）
    返回：提取出的关键条款摘要。
    """
    if _llm is None:
        return "错误：大模型未初始化。"

    # 限制输入长度，防止超上下文
    truncated = contract_text[:3000]
    prompt = f"请从以下合同中提取所有与法律责任、义务、权利相关的关键条款（如违约责任、保密义务、争议解决、终止条件、免责条款等），用简洁的列表形式输出每条的核心内容：\n{truncated}"
    response = _llm.invoke([HumanMessage(content=prompt)])
    return response.content