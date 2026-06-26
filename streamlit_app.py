# streamlit_app.py - 支持 Agent 审计 + 文件上传
import streamlit as st
import requests
import json

st.set_page_config(page_title="智能审计决策系统", layout="wide")
st.title("🔍 智能审计决策系统")
st.markdown("上传合同文档，AI 自动生成审计报告")

uploaded_file = st.file_uploader("📎 上传合同文件", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    if st.button("🚀 开始审计"):
        with st.spinner("正在分析合同..."):
            files = {"file": uploaded_file}
            try:
                response = requests.post(
                    "http://localhost:8000/agent/audit_file",
                    files=files
                )
                if response.status_code == 200:
                    result = response.json()
                    st.markdown("### 📋 审计报告")
                    st.markdown(result.get("answer", "无返回结果"))
                else:
                    st.error(f"请求失败: {response.text}")
            except Exception as e:
                st.error(f"连接失败: {e}")