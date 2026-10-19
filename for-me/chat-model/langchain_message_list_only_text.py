from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

messages = [
  SystemMessage("Please pretend to be a human"), 
  HumanMessage("I'm Makoto, Japanese."), 
  AIMessage("Hello"), 
  HumanMessage("Could you answer my name and nationality?")
]

response = llm.invoke(messages)
print(response.text)