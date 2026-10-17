from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

for chunk in llm.stream("Hello"):
  print(chunk.text, end="", flush=True)

print()