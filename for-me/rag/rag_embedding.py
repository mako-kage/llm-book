from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
import sys

collection_name = "example_collection"
persist_directory = "./chroma_db"

filename = sys.argv[1] if len(sys.argv) >= 2 else input("ベクトル化するPDFファイルのファイル名を入力(*.pdf): ")

print(filename)

embeddings = FastEmbedEmbeddings(
  model_name="intfloat/multilingual-e5-large", 
  cache_dir="/root/.cache/fastembed_cache", 
)

vector_store = Chroma(
  collection_name=collection_name,
  embedding_function=embeddings,
  persist_directory=persist_directory,
)

# すでに埋め込みベクトル作成済みのため、コメントアウト（初回実行時 or chroma_dbを消した場合は、コメントアウト外して再度実行）
# 処理長い & CPU/メモリ消費がすごいので注意
## loader = PyPDFLoader(filename)
## pages = loader.load()
## python_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
## splits = python_splitter.split_documents(pages)
## vector_store.add_documents(splits)

docs = vector_store.similarity_search("生成AIの最新動向は？", k=3)

for doc in docs: 
  print("---")
  print(doc.page_content[:100])
