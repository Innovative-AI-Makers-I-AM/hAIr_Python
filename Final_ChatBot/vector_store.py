'''벡터 검색 엔진을 다루는 함수를 제공'''
import os  # os 모듈 임포트
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma

# 문서 리스트를 받아 벡터 검색 엔진을 생성하고 각 문서를 벡터화하여 저장
def create_vector_store(docs, model_name, persist_directory, collection_metadata):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory, collection_metadata=collection_metadata)

    # 문서 개수 출력
    print(f"Number of documents create stored: {db._collection.count()}")

    return db

# 이미 생성된 벡터 검색 엔진을 불러온다.
def load_vector_store(persist_directory, model_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 문서 개수 출력
    print(f"Number of documents load stored: {db._collection.count()}")
    
    return db