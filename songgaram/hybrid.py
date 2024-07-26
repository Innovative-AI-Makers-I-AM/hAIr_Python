import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torchvision import transforms
import zipfile
import io
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.api.models.Collection import Collection
import numpy as np
from langchain.retrievers import EnsembleRetriever, BM25Retriever, TimeWeightedVectorStoreRetriever
import os

# 텍스트 임베딩 모델 로드
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
text_model.eval()

# 이미지 임베딩 모델 로드
image_model_name = "metadome/face_shape_classification"
image_model = AutoModel.from_pretrained(image_model_name)
image_model.eval()

# 이미지 전처리 함수
def preprocess_image(image_data):
    input_image = Image.open(io.BytesIO(image_data))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(input_image).unsqueeze(0)

# 텍스트 전처리 및 임베딩 추출 함수
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

# ChromaDB 초기화 함수
def initialize_chromadb(collection_names=['hair_description', 'hair_image']):
    client = chromadb.PersistentClient()
    collections = {}
    for name in collection_names:
        if name in [coll.name for coll in client.list_collections()]:
            collections[name] = client.get_collection(name=name)
            print(f'존재하는 콜렉션 사용: {name}')
        else:
            collections[name] = client.create_collection(name=name)
            print(f'새로운 콜렉션 생성 : {name}')
    return collections

collection = initialize_chromadb()

# # 데이터 추가 함수
# def add_data_to_collection(collection, data_id, embedding, metadata):
#     collection.add(
#         ids=[data_id], 
#         embeddings=[embedding.tolist()], 
#         metadatas=[metadata]
#     )

# # 데이터 예시 추가 (텍스트와 이미지 임베딩)
# add_data_to_collection(collection, "example_text", get_text_embedding("This is a sample text"), {"type": "text"})

# # 이미지 파일 경로 수정
# image_file_path = "path_to_your_image.jpg"
# if os.path.exists(image_file_path):
#     image_data = open(image_file_path, "rb").read()
#     image_embedding = image_model(preprocess_image(image_data)).last_hidden_state.mean(dim=1).numpy().flatten()
#     add_data_to_collection(collection, "example_image", image_embedding, {"type": "image"})
# else:
#     print(f"Image file '{image_file_path}' not found.")

# DenseRetriever 초기화 함수
def initialize_bm25_retriever(collection, embedding_model):
    documents = collection.get(include=["documents"])["documents"]
     # 문서 데이터를 사전 형식으로 변환
    docs = [{"text": doc} for doc in documents]
    return BM25Retriever(
        embedding_model=embedding_model,
        collection=collection,
        docs=docs
    )


# DenseRetriever 초기화 함수
def initialize_dense_retriever(collection, embedding_model):
    return TimeWeightedVectorStoreRetriever(
        embedding_model=embedding_model,
        collection=collection
    )


# 하이브리드 검색 함수 (EnsembleRetriever 사용)
def hybrid_search_with_ensemble(collections, text_query=None, image_data=None):
    text_dense_retriever = initialize_dense_retriever(collections['hair_description'], text_model)
    text_bm25_retriever = initialize_bm25_retriever(collections['hair_description'])
    image_dense_retriever = initialize_dense_retriever(collections['hair_image'], image_model)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[text_dense_retriever, text_bm25_retriever, image_dense_retriever],
        weights=[0.33, 0.33, 0.34]
    )

    results = []

    # 텍스트 쿼리 임베딩 추출 및 검색
    if text_query:
        text_embedding = get_text_embedding(text_query)
        text_results = ensemble_retriever.retrieve(embeddings=[text_embedding.tolist()])
        results.extend(text_results["documents"])

    # 이미지 쿼리 임베딩 추출 및 검색
    if image_data:
        image_embedding = image_model(preprocess_image(image_data)).last_hidden_state.mean(dim=1).numpy().flatten()
        image_results = ensemble_retriever.retrieve(embeddings=[image_embedding.tolist()])
        results.extend(image_results["documents"])

    return results

# 사용자 입력 받아서 하이브리드 검색 수행 (EnsembleRetriever 사용)
def user_input_hybrid_search_with_ensemble(collections):
    text_query = input("Enter text query: ")
    image_file_path = input("Enter path to your image file: ")

    image_data = None
    if os.path.exists(image_file_path):
        with open(image_file_path, "rb") as f:
            image_data = f.read()
    else:
        print(f"Image file '{image_file_path}' not found.")

    search_results = hybrid_search_with_ensemble(collections, text_query=text_query, image_data=image_data)

    # 검색 결과 출력
    for result in search_results:
        print(result)

# 사용자 입력을 받아서 검색 수행 (EnsembleRetriever 사용)
user_input_hybrid_search_with_ensemble(collection)
