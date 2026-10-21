from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate,
)

chat_template = ChatPromptTemplate.from_messages(
  [
    SystemMessagePromptTemplate.from_template("あなたの名前は{ai_name}です。"),
    HumanMessagePromptTemplate.from_template("私の名前は{human_name}です。"),
    AIMessage("こんにちは。"),
    HumanMessagePromptTemplate.from_template("{input}"),
  ]
)

prompt = chat_template.invoke(
  {
    "ai_name": "SAL 9000", 
    "human_name": "ヘイウッド", 
    "input": "私の名前はわかりますか？", 
  }
)

print(prompt)

llm = ChatGoogleGenerativeAI(
  model="gemini-3.5-flash-lite", 
  max_retries=0 # 輻輳時のリトライによる、RPD上限突破を防ぐ
)
response = llm.invoke(prompt)
print(response.text)

