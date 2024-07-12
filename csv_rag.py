import csv
from sentence_transformers import SentenceTransformer, util
from chromadb import PersistentClient

from langchain_community.vectorstores import Chroma
from langchain_community.llms import _import_openai
from langchain.chains import retrieval, combine_documents
from langchain.prompts import PromptTemplate


# 모델 및 ChromaDB 설정 함수
def initialize_model():
  # 모델 생성
  return SentenceTransformer('upskyy/kf-deberta-multitask')

def initialize_chromadb(collection_name='hairstyle_embeddings'):
    client = PersistentClient()
    collections = client.list_collections()
    collection_names = [coll.name for coll in collections]

    if collection_name in collection_names:
        collection = client.get_collection(name=collection_name)
        print(f'기존 컬렉션 {collection_name}을 사용합니다.')
    else:
        collection = client.create_collection(name=collection_name)
        print(f'새 컬렉션 {collection_name}을 생성했습니다.')

    return collection

# CSV 파일 로드 함수
def load_csv(file_path):
    hairstyles = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            hairstyles.append(row)
    return hairstyles

# 텍스트 임베딩 및 데이터 추가 함수
def embed_and_add_to_chromadb(model, collection, hairstyles):
    texts = [f"{hairstyle['style']}, {hairstyle['image_url']}, {hairstyle['hashtags']}" for hairstyle in hairstyles]
    embeddings = model.encode(texts)
    ids = [str(i) for i in range(len(embeddings))]
    metadatas = [{'style': hairstyle['style'], 'image_url': hairstyle['image_url'], 'hashtags': hairstyle['hashtags']} for hairstyle in hairstyles]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print('\n 임베딩 데이터 추가')
    return embeddings, texts

# 데이터 확인 함수
def print_stored_data(collection):
    stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
    print("\n hairstyle_embeddings 컬렉션에 저장된 모든 데이터:")
    for i in range(len(stored_data['ids'])):
        print(f"ID: {stored_data['ids'][i]}")
        print(f"Document: {stored_data['documents'][i]}")
        print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
        print(f"Metadata: {stored_data['metadatas'][i]}")
        print("---")

# 리트리버 초기화
def initialize_retriever(collection, model):
    class EmbeddingFunction:
        def __init__(self, model):
            self.model = model

        def embed_query(self, query):
            return self.model.encode([query])[0].tolist()

    embedding_function = EmbeddingFunction(model)
    vectorstore = Chroma(collection_name='hairstyle_embeddings', embedding_function=embedding_function)
    return vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})

# 검색 함수
def find_most_similar(query, retriever):
    retrieved_docs = retriever.get_relevant_documents(query=query)
    if retrieved_docs:
        return retrieved_docs[0].metadata
    else:
        return None

if __name__ == "__main__":
    # 모델 및 ChromaDB 초기화
    model = initialize_model()
    collection = initialize_chromadb()

    # CSV 파일 로드
    file_path = 'hairstyles.csv'
    hairstyles = load_csv(file_path)
    print(f'문서의 수: {len(hairstyles)}')

    # 텍스트 임베딩 및 ChromaDB에 데이터 추가
    embeddings, texts = embed_and_add_to_chromadb(model, collection, hairstyles)

    # 컬렉션에 저장된 데이터 확인
    print_stored_data(collection)

    # 리트리버 초기화
    retriever = initialize_retriever(collection,model)

    # 예제 쿼리 검색
    query = "레이어드컷"
    most_similar_hairstyle = find_most_similar(query, retriever)
    if most_similar_hairstyle:
        print(f"가장 유사한 스타일: {most_similar_hairstyle}")
    else:
        print(f"{query}와 유사한 스타일을 찾을 수 없습니다.")

    # RAG 설정 및 답변 생성
    llm = _import_openai()
    qa_prompt = PromptTemplate(input_variables=["context", "question"], template="Context: {context}\n\nQuestion: {question}\n\nAnswer:")
    combine_docs_chain = combine_documents.create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = retrieval.create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

    # 질문에 대한 답변 생성
    question = "레이어드컷에 대해 알려줘"
    if most_similar_hairstyle:
      result = qa_chain.invoke({"input": {"context": most_similar_hairstyle, "question": question}})
      print(f"\nAnswer: {result['result']}")
    else:
        print(f"{question}의 질문과 유사한 스타일을 찾을 수 없어 답변을 생성할 수 없습니다.")



# model = SentenceTransformer('upskyy/kf-deberta-multitask')

# # ChromaDB 설정 및 클라이언트 생성
# client = PersistentClient()

# # 컬렉션 생성 or 기존 컬렉션 가져오기
# collection_name = 'hairstyle_embeddings'
# collections = client.list_collections()
# collection_names = [coll.name for coll in collections]

# if collection_name in collection_names:
#     collection = client.get_collection(name=collection_name)
#     print(f'기존 컬렉션 {collection_name}을 사용합니다.')
# else:
#     collection = client.create_collection(name=collection_name)
#     print(f'새 컬렉션 {collection_name}을 생성했습니다.')

# # CSV 파일 로드 및 UTF-8로 읽기
# file_path = 'hairstyles.csv'
# hairstyles = []

# with open(file_path, mode='r', encoding='utf-8') as csvfile:
#     csv_reader = csv.DictReader(csvfile)
#     for row in csv_reader:
#         hairstyles.append(row)

# print(f'문서의 수: {len(hairstyles)}')

# # 텍스트 데이터 임베딩
# texts =[f"{hairstyle['style']}, {hairstyle['image_url']}, {hairstyle['hashtags']}" for hairstyle in hairstyles]
# embeddings = model.encode(texts)

# # 임베딩 된 데이터 추가
# ids = [str(i) for i in range(len(embeddings))]
# metadatas = [{'style':hairstyle['style'], 'image_url':hairstyle['image_url'], 'hashtags':hairstyle['hashtags']} for hairstyle in hairstyles]
# collection.add(
#     documents=texts,
#     embeddings=embeddings,
#     metadatas=metadatas,
#     ids=ids,
# )

# print('\n 임베딩 데이터 추가')    # 컬렉션에 저장된 값 확인
# print("\n hairstyle_embeddings 컬렉션에 저장된 모든 데이터:")
# stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
# for i in range(len(stored_data['ids'])):
#     print(f"ID: {stored_data['ids'][i]}")
#     print(f"Document: {stored_data['documents'][i]}")
#     print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
#     print(f"Metadata: {stored_data['metadatas'][i]}")
#     print("---")

# # 검색 함수 정의
# def find_most_similar(query, embeddings, texts, data):
#     query_embedding = model.encode(query)
#     similarities = util.pytorch_cos_sim(query_embedding, embeddings)
#     most_similar_index = similarities.argmax().item()
#     return data[most_similar_index]

# # 예제 쿼리
# query = "레이어드컷"
# most_similar_hairstyle = find_most_similar(query, embeddings, texts, hairstyles)
# print(f"가장 유사한 스타일: {most_similar_hairstyle}")


