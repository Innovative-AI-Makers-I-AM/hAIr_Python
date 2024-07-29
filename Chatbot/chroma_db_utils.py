import chromadb
from chromadb.config import Settings

# Chroma DB 초기화 함수
def initialize_chroma_db(persist_directory):
    client = chromadb.PersistentClient(path=persist_directory)
    return client

# Chroma DB에 임베딩 데이터 추가
def add_embeddings_to_chroma(client, collection_name, embeddings, metadata, ids):
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )

# 임베딩 데이터 검색 함수
def search_chroma(client, collection_name, query_embedding, k=5):
    collection = client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results

# ID로 데이터를 조회하는 함수
def get_data_by_id(client, collection_name, embedding_id):
    collection = client.get_collection(name=collection_name)
    results = collection.get(
        ids=[embedding_id],
        include=["embeddings", "metadatas", "documents"]
    )
    return results

# ㅁ
def get_all_ids_from_chroma(client, collection_name):
    try:
        collection = client.get_or_create_collection(name=collection_name)
        ids = collection.get()["ids"]
        return ids
    except Exception as e:
        print(f"Error getting IDs from Chroma DB: {e}")
        return []


def search_chroma_with_filter(chroma_client, collection_name, query_embedding, filter_conditions, k=5):
    collection = chroma_client.get_collection(name=collection_name)
    
    if len(filter_conditions) > 1:
        # 필터 조건이 두 개 이상일 때만 AND 조건으로 결합
        filter_query = { '$and': [ {key: value} for key, value in filter_conditions.items() ] }
    else:
        # 필터 조건이 하나일 때는 바로 사용
        filter_query = filter_conditions
    
    print(f"Applying filter conditions: {filter_conditions}")
    
    # ChromaDB에서 여러 조건을 처리할 수 있도록 where 필터를 사용
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_query
        )
        # print(f"Query results: {results}")
    except Exception as e:
        print(f"Error during ChromaDB query: {e}")
        raise e
    
    return results