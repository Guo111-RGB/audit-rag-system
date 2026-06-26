📌 项目简介
本项目是一个基于大语言模型（LLM）的智能审计决策系统，用户上传合同文档后，系统自动识别条款风险并生成专业法律合规审计报告。

核心能力：

支持 5 种以上合同风险场景自动识别

审计报告生成时间 < 30 秒

所有数据本地化处理，满足金融审计数据安全要求

✨ 核心功能
功能	说明
📄 多格式文件上传	支持 PDF / Word / TXT 合同文档自动解析
🤖 多智能体审计	Agent 自主规划检索策略，信息不足时自动补充检索
🔍 混合检索	BM25 精确匹配 + 向量语义检索，召回率 90%+
📊 专业报告生成	合规性评估 → 风险清单 → 修改建议 → 总体结论
🔒 私有化部署	所有数据本地处理（ChromaDB + 本地嵌入模型）
🧠 自我反思机制	Agent 主动评估信息充分性，避免遗漏关键风险
🛠️ 技术栈
AI 与 LLM
LangChain · LangGraph · 通义千问 Qwen-Plus

Prompt Engineering · 工具调用 (Tool Calling)

检索与知识库
ChromaDB · BM25 · 混合检索 · BGE 嵌入模型 · Jieba 分词

后端与 API
Python 3.10+ · FastAPI · Uvicorn · Pydantic

前端与界面
Streamlit · Swagger UI

文档处理
pdfplumber · PyPDF · python-docx · 正则表达式

🚀 快速开始
1. 克隆项目
bash
git clone https://github.com/Guo111-RGB/audit-rag-system.git
cd audit-rag-system
2. 创建虚拟环境
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3. 安装依赖
bash
pip install -r requirements.txt
4. 配置 API Key
在项目根目录创建 .env 文件：

env
DASHSCOPE_API_KEY=你的通义千问API密钥
5. 构建知识库索引
bash
python scripts/build_clause_index_from_pdf.py
6. 启动服务
bash
# 终端1：启动后端
python app/main.py

# 终端2：启动前端
streamlit run streamlit_app.py
7. 访问系统
服务	地址
用户端界面	http://localhost:8501
API 文档 (Swagger)	http://localhost:8000/docs
API 文档 (ReDoc)	http://localhost:8000/redoc
📁 项目结构
text
audit-rag-system/
├── app/
│   ├── agent/              # 多智能体模块
│   │   ├── graph.py        # LangGraph 状态图
│   │   ├── tools.py        # Agent 工具定义
│   │   └── prompts.py      # 提示词模板
│   ├── hybrid_retriever.py # 混合检索器（BM25 + 向量）
│   └── main.py             # FastAPI 入口
├── data/
│   └── full_docs/          # 法律文档 PDF
├── scripts/
│   └── build_clause_index_from_pdf.py  # 索引构建脚本
├── chroma_db_clause/       # 向量数据库
├── streamlit_app.py        # 用户端界面
└── requirements.txt        # 依赖清单
🧪 测试示例
使用 Swagger UI 测试
启动服务后访问 http://localhost:8000/docs

找到 POST /agent/audit 接口

输入合同条款，点击 Execute

示例输入
json
{
  "query": "甲方有权收集乙方的姓名、身份证号、手机号、银行卡号，并可将这些信息用于市场营销推广、产品推荐以及向第三方合作机构共享，无需另行通知乙方。"
}
示例输出
text
=== 审计结论 ===
不合规

=== 风险清单 ===
1. 未经同意收集个人信息 → 违反《个人信息保护法》第四条
2. 向第三方共享信息未获授权 → 违反《个人信息保护法》第二十三条
3. 未提供拒绝营销选项 → 违反《个人信息保护法》第二十四条

=== 修改建议 ===
1. 增加用户单独同意条款
2. 明确告知信息共享对象
3. 提供一键退订功能
🤝 核心亮点
多智能体架构：基于 LangGraph 实现 ReAct 模式，系统能自主规划、执行、反思

混合检索：BM25 + 向量融合，精确率 90%+

条款级分割：解决固定分块切断法条的问题

私有化部署：数据不出本地，满足金融审计安全要求

端到端闭环：从上传文档到输出报告，完整产品链路

📄 License
MIT License

如果这个项目对你有帮助，欢迎 Star ⭐
