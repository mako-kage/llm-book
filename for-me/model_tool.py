from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

messages = [
  SystemMessage("あなたの名前はハルです。"), 
  HumanMessage("私の名前はデイブです。"), 
  AIMessage("こんにちは、デイブさん。"), 
  HumanMessage("ハル、ポッドのライトを20度左に回してくれ。"), 
]

@tool
def light_contrl(degrees: float) -> bool:
  """
  ライトを右に degrees 度回す。
  degreesは、0から360までの値を取る。
  """
  # ここでライトを右に degrees 度回す
  print(f"light_contrl: {degrees}")
  return True

llm_with_tool = llm.bind_tools([light_contrl])
response = llm_with_tool.invoke(messages)
messages.append(response)
if response.tool_calls:
  for call in response.tool_calls:
    print(call)
    value = light_contrl.invoke(call["args"])
    messages.append(ToolMessage(value, tool_call_id=call["id"]))

response = llm_with_tool.invoke(messages)
print(response)
print(messages)