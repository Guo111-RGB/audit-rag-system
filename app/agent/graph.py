import operator
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.agent.tools import search_law, extract_clauses, set_llm

# 全局 LLM 实例
llm = None


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_action: str
    tool_results: List[str]
    step_count: int  # 新增：步骤计数器


tools = [search_law, extract_clauses]

SYSTEM_PROMPT = """你是一名金融审计与法律合规专家。
你可以使用以下工具：
- search_law: 检索法律法规
- extract_clauses: 从合同中提取关键条款

请根据用户的问题，自主决定需要调用哪些工具、调用顺序以及是否需要重复调用。
在调用工具后，观察返回结果，如果信息不足可以再次调用工具。
当你认为信息足够时，直接回答用户的问题，生成审计报告。
回答要专业、简洁，并引用具体法条编号。
【重要限制】：
1. 只分析用户提供的合同条款原文中明确列出的内容，不要编造或假设不存在的条款。
2. 引用法条时，必须来自你检索到的【法律法规原文】，不得引用未检索到的法律。
3. 如果合同中没有提到某项内容（如定金），不要凭空分析。"""


def agent_node(state: AgentState):
    global llm
    if llm is None:
        raise ValueError("LLM 未初始化")

    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm.bind_tools(tools)
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}


def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    outputs = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if tool_name == "search_law":
            result = search_law.invoke(tool_args)
        elif tool_name == "extract_clauses":
            result = extract_clauses.invoke(tool_args)
        else:
            result = f"未知工具: {tool_name}"
        outputs.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState) -> Literal["tools", "reflect", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "reflect"


def reflect_node(state: AgentState):
    global llm
    if llm is None:
        raise ValueError("LLM 未初始化")

    step_count = state.get("step_count", 0)
    messages = state["messages"]

    # 如果步骤超过2次，强制生成报告并结束
    if step_count >= 2:
        info = "\n".join([
            m.content for m in messages
            if isinstance(m, (AIMessage, ToolMessage))
        ])
        report_prompt = f"""请基于以下信息生成一份完整的审计报告，引用具体法条：
{info}
"""
        final_report = llm.invoke([HumanMessage(content=report_prompt)]).content
        return {
            "messages": [AIMessage(content=final_report)],
            "next_action": "end",
            "step_count": step_count + 1
        }

    info = "\n".join([
        m.content for m in messages
        if isinstance(m, (AIMessage, ToolMessage))
    ])

    prompt = f"""你已获得以下信息：
{info}

请评估这些信息是否足以回答用户的审计问题。
如果信息不足，请明确说明缺少什么，并给出新的检索关键词。
如果信息已充足，请回答"信息已充足"。
"""
    reflect_response = llm.invoke([HumanMessage(content=prompt)])
    answer = reflect_response.content.strip()

    if "信息已充足" in answer:
        # 直接生成最终报告并结束
        report_prompt = f"""请基于以下信息生成一份完整的审计报告，引用具体法条：
{info}
"""
        final_report = llm.invoke([HumanMessage(content=report_prompt)]).content
        return {
            "messages": [AIMessage(content=final_report)],
            "next_action": "end",
            "step_count": step_count + 1
        }
    else:
        return {
            "messages": [HumanMessage(content=f"需要补充检索：{answer}")],
            "next_action": "retry",
            "step_count": step_count + 1
        }

def should_after_reflect(state: AgentState) -> Literal["agent", "__end__"]:
    action = state.get("next_action", "")
    if action in ["generate", "retry"]:
        return "agent"
    return "__end__"


def build_agent_graph(llm_instance):
    global llm
    llm = llm_instance
    set_llm(llm_instance)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reflect", reflect_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "reflect": "reflect",
            "__end__": END
        }
    )
    workflow.add_edge("tools", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_after_reflect,
        {
            "agent": "agent",
            "__end__": END
        }
    )

    return workflow.compile()