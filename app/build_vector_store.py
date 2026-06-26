# app/build_vector_store.py（增强版，支持多种审计知识结构）
import os
import json
import shutil
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "audit_standards.json")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def load_standards_from_json():
    """智能加载 JSON 中的各种结构，统一转换成 Document 列表"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for std_name, std_content in data.items():
        print(f"处理标准: {std_name}")

        # 1. COBIT 2019 风格：domains -> domain -> control_objectives
        if "domains" in std_content:
            for domain_code, domain_info in std_content["domains"].items():
                # 兼容两种：domain_info 可能是 dict 包含 control_objectives，也可能是直接列表（如 CISA 的某些域）
                if isinstance(domain_info, dict):
                    # COBIT 风格：domain_info 里有 "full_name" 和 "control_objectives"
                    domain_fullname = domain_info.get("full_name", domain_code)
                    objs = domain_info.get("control_objectives", [])
                    for obj in objs:
                        content = f"【标准】{std_name}  【领域】{domain_code} - {domain_fullname}  【控制目标】{obj['id']}: {obj['name']}\n{obj['description']}"
                        metadata = {"source": std_name, "type": "control_objective", "id": obj['id']}
                        documents.append(Document(page_content=content, metadata=metadata))
                elif isinstance(domain_info, list):
                    # CISA 风格：domain 的值直接是一个列表，列表元素有 id,name,description
                    for item in domain_info:
                        if "id" in item and "name" in item:
                            content = f"【标准】{std_name}  【类别】{domain_code}  【要点】{item['id']}: {item['name']}\n{item.get('description', '')}"
                            metadata = {"source": std_name, "type": "cisa_knowledge", "id": item['id']}
                            documents.append(Document(page_content=content, metadata=metadata))

        # 2. ISO 27001 风格：annex_a 列表
        elif "annex_a" in std_content:
            for ctrl in std_content["annex_a"]:
                content = f"【标准】{std_name}  【控制措施】{ctrl['id']}: {ctrl['name']}\n{ctrl.get('description', '')}"
                metadata = {"source": std_name, "type": "control", "id": ctrl['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 3. SOX 风格：sections 列表
        elif "sections" in std_content:
            for sec in std_content["sections"]:
                content = f"【标准】{std_name}  【条款】{sec['id']}: {sec['name']}\n{sec.get('description', '')}"
                metadata = {"source": std_name, "type": "section", "id": sec['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 4. 民法典风格：parts -> chapters
        elif "parts" in std_content:
            for part in std_content["parts"]:
                part_name = part.get("part", "")
                for chapter in part.get("chapters", []):
                    content = f"【标准】{std_name}  【章节】{chapter['id']}: {chapter['name']}\n{chapter.get('description', '')}"
                    metadata = {"source": std_name, "type": "law", "id": chapter['id']}
                    documents.append(Document(page_content=content, metadata=metadata))

        # 5. COSO 2013 风格：domains 下的每个 domain 包含 principles 列表
        elif "description" in std_content and "domains" in std_content:
            # 处理 COSO 结构： domains -> domain -> principles
            for domain_name, domain_info in std_content["domains"].items():
                principles = domain_info.get("principles", [])
                for p in principles:
                    content = f"【标准】{std_name}  【领域】{domain_name}  【原则】{p['id']}: {p['name']}\n{p['description']}"
                    metadata = {"source": std_name, "type": "principle", "id": p['id']}
                    documents.append(Document(page_content=content, metadata=metadata))

        # 6. GAAP_IFRS 风格：sections 列表（直接顶层）
        elif "sections" in std_content and isinstance(std_content["sections"], list):
            for sec in std_content["sections"]:
                content = f"【标准】{std_name}  【准则】{sec['id']}: {sec['name']}\n{sec.get('description', '')}"
                metadata = {"source": std_name, "type": "accounting", "id": sec['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 7. 中国数据安全法规：laws 列表
        elif "laws" in std_content:
            for law in std_content["laws"]:
                content = f"【标准】{std_name}  【法规】{law['id']}: {law['name']}\n{law['description']}"
                metadata = {"source": std_name, "type": "law", "id": law['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 8. 行业标准示例：standards 列表
        elif "standards" in std_content:
            for std in std_content["standards"]:
                content = f"【标准】{std_name}  【行业标准】{std['id']}: {std['name']}\n{std['description']}"
                metadata = {"source": std_name, "type": "industry", "id": std['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 9. 敏捷审计理念：concepts 列表
        elif "concepts" in std_content:
            for concept in std_content["concepts"]:
                content = f"【标准】{std_name}  【理念】{concept['id']}: {concept['name']}\n{concept['description']}"
                metadata = {"source": std_name, "type": "methodology", "id": concept['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 10. 风险控制矩阵 RCM：cycles 列表
        elif "cycles" in std_content:
            for cycle in std_content["cycles"]:
                cycle_name = cycle.get("cycle", "")
                for risk in cycle.get("risks", []):
                    controls_str = "；".join(risk.get("controls", []))
                    content = f"【标准】{std_name}  【业务循环】{cycle_name}  【风险】{risk['id']}: {risk['name']}\n控制措施：{controls_str}"
                    metadata = {"source": std_name, "type": "rcm", "id": risk['id']}
                    documents.append(Document(page_content=content, metadata=metadata))

        # 11. 中国审计准则：standards 列表（与行业标准类似）
        elif "standards" in std_content and std_name == "China_Audit_Standards":
            for std in std_content["standards"]:
                content = f"【标准】{std_name}  【审计准则】{std['id']}: {std['name']}\n{std['description']}"
                metadata = {"source": std_name, "type": "auditing_standard", "id": std['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 12. 审计程序库：procedures 列表
        elif "procedures" in std_content:
            for proc in std_content["procedures"]:
                content = f"【标准】{std_name}  【审计程序】{proc['id']}: {proc['name']}\n{proc['description']}"
                metadata = {"source": std_name, "type": "procedure", "id": proc['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 13. 常见发现与整改：findings 列表
        elif "findings" in std_content:
            for finding in std_content["findings"]:
                content = f"【标准】{std_name}  【审计发现】{finding['id']}: {finding['finding']}\n严重程度：{finding['severity']}\n整改建议：{finding['remediation']}"
                metadata = {"source": std_name, "type": "finding", "id": finding['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        # 14. 数据分析规则：rules 列表
        elif "rules" in std_content:
            for rule in std_content["rules"]:
                content = f"【标准】{std_name}  【分析规则】{rule['id']}: {rule['name']}\n{rule['description']}"
                metadata = {"source": std_name, "type": "caats", "id": rule['id']}
                documents.append(Document(page_content=content, metadata=metadata))

        else:
            print(f"警告：无法识别的结构 - {std_name}，跳过")

    print(f"✓ 共加载 {len(documents)} 条知识条目")
    return documents


def build_vector_store():
    print("正在加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("正在加载标准数据...")
    docs = load_standards_from_json()

    # 删除旧数据库
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("已删除旧数据库")

    print(f"总计 {len(docs)} 个文档块，正在创建向量数据库...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"✓ 向量数据库已保存到: {CHROMA_PATH}")
    print("✓ 构建完成！")


if __name__ == "__main__":
    build_vector_store()
    build_vector_store()