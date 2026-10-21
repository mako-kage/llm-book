from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import os

url = "https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r05/pdf/00zentai.pdf"
filename = "情報通信白書.pdf"

if not os.path.exists(filename):
  with open(filename, "wb") as file:
    file.write(requests.get(url).content)

loader = PyPDFLoader(filename)
pages = loader.load()
python_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
splits = python_splitter.split_documents(pages)
print(f"ページ数: {len(splits)}")
n = 100
print(f"{n}チャンク目： {splits[n].page_content[:100]}")