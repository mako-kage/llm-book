from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

messages = llm.batch(
  [
    "Good morning", 
    "Hello", 
    "Good evening", 
  ]
)

for message in messages:
  print(message.text)