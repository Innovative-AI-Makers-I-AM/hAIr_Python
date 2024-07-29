import os  # os 모듈 임포트
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(docs, model_name, persist_directory, collection_metadata):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory, collection_metadata=collection_metadata)

    # 문서 개수 출력
    print(f"Number of documents create stored: {db._collection.count()}")

    return db

def load_vector_store(persist_directory, model_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 문서 개수 출력
    print(f"Number of documents load stored: {db._collection.count()}")
    
    return db