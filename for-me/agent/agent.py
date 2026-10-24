from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
  ChatPromptTemplate,
  MessagesPlaceholder, 
  HumanMessagePromptTemplate,
)
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from typing import Literal

angle = 50

@tool 
def get_current_light_angle() -> float:
  """
  現在のライトの角度（0 <= degree < 360）を返す。
  """
  return angle

@tool 
def light_control(direction: Literal["right", "left"]) -> bool:
  """
  ライトをdirectionに少しだけ回す。
  成功した場合は True を返す。
  回転後の角度はget_current_light_angle()で取得可能。
  """
  global angle
  angle += 10 if direction == "right" else -10
  angle += 360 if angle < 0 else 0
  return True

tools = [get_current_light_angle, light_control]

prompt = ChatPromptTemplate.from_messages(
  [
    SystemMessage(
      """
      あなたの名前はハルです。
      あなたの仕事は宇宙船の制御です。
      ツール呼び出しごとに計器を確認して下さい。
      必ずCoT推論を行ってからツール呼び出しを行ってください。
      推論の過程も必ず示してください。
      """
    ), 
    MessagesPlaceholder("chat_history"), 
    HumanMessagePromptTemplate.from_template("{input}"), 
    MessagesPlaceholder("agent_scratchpad"), 
  ]
)

llm = ChatGoogleGenerativeAI(
  model="gemini-3.5-flash-lite", 
  max_retries=0,  # 輻輳時のリトライによる、RPD上限突破を防ぐ
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
answer = agent_executor.invoke(
  {"input": "ハル、ポッドのライトを20度左に回してくれ。", "chat_history": []}
)
print("回答：", answer["output"])