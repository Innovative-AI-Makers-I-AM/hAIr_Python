'''텍스트 파일을 읽어 분할하는 함수를 제공'''
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 함수는 지정된 디렉토리에서 텍스트 파일을 읽어 분할된 문서 리스트를 반환
def load_and_split_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            print(f"Loading file: {file}")  # 파일 로딩 확인
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")
            loaded_docs = loader.load_and_split()
            print(f"Loaded {len(loaded_docs)} documents from {file}")  # 로드한 문서 개수 확인
            documents.extend(loaded_docs)
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function = len)
    split_docs = text_splitter.split_documents(documents)
    print(f"Total number of split documents: {len(split_docs)}")  # 스플릿한 문서 개수 확인


    return split_docs
