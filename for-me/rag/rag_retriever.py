from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

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
docs = retriever.invoke("生成AIの最新動向")

for doc in docs: 
  print("---")
  print(doc.page_content[:100])
