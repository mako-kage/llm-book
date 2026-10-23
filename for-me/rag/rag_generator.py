from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
  ChatPromptTemplate,
  MessagesPlaceholder, 
  HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from operator import itemgetter

collection_name = "example_collection"
persist_directory = "./chroma_db"

embeddings = FastEmbedEmbeddings(
  model_name="intfloat/multilingual-e5-large", 
  cache_dir="/root/.cache/fastembed_cache", 
)

vector_store = Chroma(
  collection_name=collection_name,
  embedding_function=embeddings,
  persist_directory=persist_directory,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages(
  [
    SystemMessage("あなたは有能なアシスタントです。"), 
    MessagesPlaceholder("chat_history"), 
    HumanMessagePromptTemplate.from_template(
      "与えられた文脈に基づいて、次の質問に答えてください。\n" \
      "文脈：{context}\n" \
      "質問：{question}"
    ), 
  ]
)

llm = ChatGoogleGenerativeAI(
  model="gemini-3.5-flash-lite", 
  max_retries=0,  # 輻輳時のリトライによる、RPD上限突破を防ぐ
  templature=0,  # 生成のランダム性を抑制
)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
  {
    "context": itemgetter("question") | retriever | format_docs, 
    "question": itemgetter("question"), 
    "chat_history": itemgetter("chat_history"), 
  }
  | prompt_template 
  | llm 
  | StrOutputParser()
)

history = []
answer = rag_chain.invoke({"question": "生成AIの最新動向は？", "chat_history": history})
print(answer)