from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(
  model="gemini-3.5-flash-lite", 
  max_retries=0 # 輻輳時のリトライによる、RPD上限突破を防ぐ
)

class Language(BaseModel):
  language_name: str = Field(description="言語名(e.g. 'Japanese')")

llm_with_language_output = llm.with_structured_output(Language)

ask_language_prompt = PromptTemplate.from_template(
  "以下の文章が書かれている言語の名前は何ですか？\n"
  "```\n"
  "{input}\n"
  "```"
)

get_language_chain = ask_language_prompt | llm_with_language_output

translation_prompt = PromptTemplate.from_template(
  "次の文章を{language}に翻訳し、"
  "翻訳された文章だけを答えてください。\n"
  "```\n"
  "{input}\n"
  "```"
)

translation = translation_prompt | llm | StrOutputParser()

to_english = {
  "input": RunnablePassthrough(),
  "language": lambda _: "English", 
} | translation

chain = {
  "input": to_english | llm | StrOutputParser(), 
  "language": get_language_chain | (lambda x: x.language_name), 
} | translation

text = input("User: ")
answer = chain.invoke(text)

print("AI: ", answer)