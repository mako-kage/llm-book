from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

message = AIMessage("こんにちは。")
s = "こんばんは。"

parser = StrOutputParser()
result1 = parser.invoke(message)
result2 = parser.invoke(s)

print(message)
print(result1)
print("=====")
print(s)
print(result2)

# llm = ChatGoogleGenerativeAI(
#   model="gemini-3.5-flash-lite", 
#   max_retries=0 # 輻輳時のリトライによる、RPD上限突破を防ぐ
# )