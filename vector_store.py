from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(docs, model_name, persist_directory, collection_metadata):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory, collection_metadata=collection_metadata)
    return db

def load_vector_store(persist_directory, model_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db