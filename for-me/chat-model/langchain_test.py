from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

response = llm.invoke("Who are you?")
print(response.content)