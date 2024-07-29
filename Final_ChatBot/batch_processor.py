'''여러 개의 문서를 한 번에 벡터화하는 작업을 쉽게 해주는 함수를 제공'''
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, load_vector_store

# 지정된 경로에서 텍스트 파일을 읽어 각 파일을 작은 문서 단위로 분할
# 분할된 문서들은 벡터 검색 엔진에 저장되며, 이미 저장된 벡터 데이터베이스가 있으면 새로운 문서만 추가
def batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata):
    
    # 지정된 경로에서 `.txt` 확장자를 가진 파일 목록을 가져옵니다.
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    # 파일 목록의 총 개수(길이)를 반환
    total_files = len(files)

    
    for i in range(0, total_files, batch_size):
        # i번째 부터 i+batch_size 까지의 파일 목록을 불러온다.
        batch_files = files[i:i+batch_size]

        print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        
        documents = []
        
        # batch_files에 담긴 파일을 하나씩 처리
        for file in batch_files:
            # 현재 처리중인 파일 이름 출력
            print(f"Loading file: {file}")
            # TextLoader를 사용해서 파일을 읽어오고 loader 변수에 저장
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")
            # 하나의 파일에 여러 개의 문서가 포함되어 있다면 각 문서를 별도의 항목으로 분리 일반적으로 줄 바꿈으로 문서를 판단
            loaded_docs = loader.load_and_split()
            # 해당 파일에 분할된 문서의 개수 출력
            print(f"Loaded {len(loaded_docs)} documents from {file}")
            # 분리된 문서들을 다시 하나로 병함
            documents.extend(loaded_docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function=len)
        
        split_docs = text_splitter.split_documents(documents)
        
        print(f"Total number of split documents: {len(split_docs)}")
        
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            db = create_vector_store(split_docs, model_name, persist_directory, collection_metadata)
        else:
            db = load_vector_store(persist_directory, model_name)
            db.add_documents(split_docs)