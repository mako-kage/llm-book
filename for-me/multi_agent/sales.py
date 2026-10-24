from typing import Annotated, TypedDict, Sequence
import operator
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langgraph.graph import END, StateGraph
import functools


salesman_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "あなたは熱意ある壺のベテラン訪問販売員、坪田壺夫です。"
            "営業が終了したら、「FINISH」と回答してください。"
            "壺が売れるか、売れる見込みがない場合に営業を終了してください。"
        ),
        HumanMessage(content="こんにちは。どちら様でしょうか？"),
        ("placeholder", "{messages}"),
    ]
)

shed_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("あなたは堅実な主夫の堅木実です。"),
        ("placeholder", "{messages}"),
    ]
)

llm = ChatGoogleGenerativeAI(
  model="gemini-3.5-flash-lite", 
  max_retries=0,  # 輻輳時のリトライによる、RPD上限突破を防ぐ
)

salesman_agent = salesman_prompt | llm
shed_agent = shed_prompt | llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state, agent, name):
    result = agent.invoke(state)
    message = HumanMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    print(f"{name}: {message.text}")
    return {"messages": [message]}


def route(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "FINISH" in last_message.text:
        return "finish"
    return "continue"


salesman_node = functools.partial(agent_node, agent=salesman_agent, name="Salesman")
shed_node = functools.partial(agent_node, agent=shed_agent, name="SHED")

workflow = StateGraph(AgentState)
workflow.add_node("Salesman", salesman_node)
workflow.add_node("SHED", shed_node)
workflow.add_conditional_edges(
    "Salesman",
    route,
    {
        "continue": "SHED",
        "finish": END,
    },
)
workflow.add_edge("SHED", "Salesman")
workflow.set_entry_point("Salesman")
graph = workflow.compile()
graph.invoke({"messages": []})
