import os
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain.text_splitter import CharacterTextSplitter

# 모델 및 ChromaDB 설정 함수
def initialize_model(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
    # 모델 생성
    return SentenceTransformer(model_name)

def initialize_chromadb(collection_name='hair_description'):
    client = PersistentClient()
    collections = client.list_collections()
    collection_names = [coll.name for coll in collections]
    print(f'collection list : {collection_names}')

    if collection_name in collection_names:
        collection = client.get_collection(name=collection_name)
        print(f'기존 컬렉션 {collection_name}을 사용합니다.')
    else:
        collection = client.create_collection(name=collection_name)
        print(f'새 컬렉션 {collection_name}을 생성했습니다.')
    
    return collection

# 텍스트 파일 읽기 및 데이터 청크 함수
def load_and_chunk_texts(folder_path, chunk_size=1000, chunk_overlap=20):
    texts = []
    filenames = []
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks = text_splitter.split_text(text)
                texts.extend(chunks)
                filenames.extend([filename] * len(chunks))  # 각 청크에 대한 파일명 저장
    
    return texts, filenames

# 텍스트 임베딩 및 데이터 추가 함수
def embed_and_add_to_chromadb(model, collection, texts, filenames):
    embeddings = model.encode(texts)
    ids = [str(i) for i in range(len(embeddings))]
    metadatas = [{'filename': filenames[i], 'id': ids[i]} for i in range(len(filenames))]
    
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    print('\n 임베딩 데이터 추가 완료')
    return embeddings, texts

# 데이터 확인 함수
def display_embeddings(collection):
    stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
    id_count = collection.count()
    print(f"총 저장된 문서 수: {id_count}")
    print("\n컬렉션에 저장된 모든 데이터:")
    for i in range(len(stored_data['ids'])):
        print(f"ID: {stored_data['ids'][i]}")
        print(f"Document: {stored_data['documents'][i][:100]}...")  # 일부만 출력
        print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
        print(f"Metadata: {stored_data['metadatas'][i]}")
        print("---")

# 실행 로직
if __name__ == "__main__":
    # 모델 및 컬렉션 초기화
    embedding_model = initialize_model()
    collection = initialize_chromadb(collection_name="hair_description")

    # 텍스트 파일 읽기 및 임베딩 생성
    texts, filenames = load_and_chunk_texts("docs")
    embed_and_add_to_chromadb(embedding_model, collection, texts, filenames)

    # 저장된 임베딩 값을 출력
    display_embeddings(collection)

    # 컬렉션이 잘 생성되었는지 확인
    print("생성된 컬렉션 목록:", [coll.name for coll in PersistentClient().list_collections()])
