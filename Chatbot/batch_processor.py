import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, load_vector_store

def batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata):
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    total_files = len(files)
    for i in range(0, total_files, batch_size):
        batch_files = files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        documents = []
        for file in batch_files:
            print(f"Loading file: {file}")
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")
            loaded_docs = loader.load_and_split()
            print(f"Loaded {len(loaded_docs)} documents from {file}")
            documents.extend(loaded_docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
        split_docs = text_splitter.split_documents(documents)
        print(f"Total number of split documents: {len(split_docs)}")
        
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            db = create_vector_store(split_docs, model_name, persist_directory, collection_metadata)
        else:
            db = load_vector_store(persist_directory, model_name)
            db.add_documents(split_docs)

    