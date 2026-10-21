from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate,
  MessagesPlaceholder, 
)

chat_template = ChatPromptTemplate.from_messages(
  [
    SystemMessagePromptTemplate.from_template("あなたの名前は{ai_name}です。"), 
    MessagesPlaceholder("chat_history"), 
    HumanMessagePromptTemplate.from_template("{input}"),
  ]
)

llm = ChatGoogleGenerativeAI(
  model="gemini-3.5-flash-lite", 
  max_retries=0 # 輻輳時のリトライによる、RPD上限突破を防ぐ
)

def chat(input, history):
  messages = chat_template.invoke(
    { "ai_name": "SAL 9000", "chat_history": history, "input": input }
  )
  ai_message = llm.invoke(messages)
  history.append(HumanMessage(input))
  history.append(ai_message)

history = []
while True:
  text = input("User: ")
  chat(text, history)
  print("AI: ", history[-1].text)