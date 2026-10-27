from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import sys

collection_name = "collection_for_rag_chatbot"
persist_directory = "./data"

embeddings = FastEmbedEmbeddings(
  model_name="intfloat/multilingual-e5-large", 
  cache_dir="/root/.cache/fastembed_cache", 
)

vector_store = Chroma(
  collection_name=collection_name,
  embedding_function=embeddings,
  persist_directory=persist_directory,
)

def load_document(filename):
  loader = CSVLoader(filename, autodetect_encoding=True)
  pages = loader.load()
  python_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
  splits = python_splitter.split_documents(pages)
  vector_store.add_documents(splits)

load_document(sys.argv[1])