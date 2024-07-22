import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)