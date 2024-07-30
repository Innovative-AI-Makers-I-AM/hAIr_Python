import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, load_vector_store

# 주어진 디렉토리의 텍스트 파일을 배치 단위로 처리하여, 문서를 로드하고 분할한 후, 
# 벡터 스토어에 저장하거나 추가하는 작업을 수행
def batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata):
    
    # 데이터 디렉토리에서 .txt 파일 목록을 불러옴
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    total_files = len(files)

    # 파일을 batch_size 크기만큼 나누어 처리
    for i in range(0, total_files, batch_size):
        batch_files = files[i:i+batch_size]  # 현재 배치의 파일 목록을 가져옵니다.
        print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")

        documents = []  # 로드된 문서를 저장할 리스트를 초기화
        
        # 배치 파일을 순회하며 문서를 로드
        for file in batch_files:
            print(f"Loading file: {file}")  # 로드할 파일명을 출력합니다.
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")  # TextLoader 인스턴스를 생성합니다.
            loaded_docs = loader.load_and_split()   # 파일을 로드하고 문서로 분할합니다.
            print(f"Loaded {len(loaded_docs)} documents from {file}")   # 로드된 문서 수를 출력합니다.
            documents.extend(loaded_docs)   # 로드된 문서를 documents 리스트에 추가합니다.
        
        # 문서를 청크 단위로 분할하기 위해 RecursiveCharacterTextSplitter를 사용
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
        split_docs = text_splitter.split_documents(documents)   # 문서를 분할합니다.
        print(f"Total number of split documents: {len(split_docs)}")    # 분할된 총 문서 수를 출력합니다.
        
         # 벡터 스토어가 존재하지 않으면 새로 생성하고, 존재하면 문서를 추가합니다.
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            # 벡터 스토어를 생성합니다.
            db = create_vector_store(split_docs, model_name, persist_directory, collection_metadata)
        else:
            # 기존 벡터 스토어를 로드하고 문서를 추가합니다.
            db = load_vector_store(persist_directory, model_name)
            db.add_documents(split_docs)
