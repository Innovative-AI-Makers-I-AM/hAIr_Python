# 얼굴분석 임베딩
# Step 1. 필요한 모듈 임포트
import torch
from transformers import AutoModel
from PIL import Image
from torchvision import transforms
import zipfile
import io
import os
import numpy as np
import chromadb

# Step 2. 추론기 생성
model_name = "metadome/face_shape_classification"
model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델 평가 모드 설정
model.eval()


# Step 3. 이미지 로드 및 전처리
def preprocess_image(image_data):
  input_image = Image.open(io.BytesIO(image_data))
  preprocess = transforms.Compose([ # 전처리 파이프라인을 설정
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),# 이미지를 텐서(다차원의 배열)로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 정규화
  ])
  return preprocess(input_image).unsqueeze(0)

# ChromaDB 초기화 함수
def initialize_chromadb(collection_name):
    client = chromadb.PersistentClient()
    collections = client.list_collections()
    collection_names = [coll.name for coll in collections]
    print(f'Collection list: {collection_names}')

    if collection_name in collection_names:
        collection = client.get_collection(name=collection_name)
        print(f'Using existing collection: {collection_name}')
    else:
        collection = client.create_collection(name=collection_name)
        print(f'Created new collection: {collection_name}')
    
    return collection

# image_tensor = preprocess_image("images/여성_단발_레이어드컷_0.jpg")

collection_name = "hair_image"
collection = initialize_chromadb(collection_name)

# Step 4. 데이터 추론(마지막 레이어 제거)
zip_file_path = "images/images_female-20240722T004917Z-001.zip"


embeddings = {}

# ZIP 파일 내에서 이미지 처리
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for image_name in zip_ref.namelist():
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_data = zip_ref.read(image_name)
            image_tensor = preprocess_image(image_data)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embedding_vector = embedding.numpy().flatten()
                
            embeddings[image_name] = embedding_vector
            collection.add(ids=[image_name], 
                embeddings=[embedding_vector.tolist()], 
                metadatas=[{"filename": image_name}])

# # 임베딩 값 출력
# for image_name, embedding_vector in embeddings.items():
#     print(f"{image_name}: {embedding_vector[:3]}")

# 임베딩된 값의 개수 출력
print(f"Number of embeddings: {len(collection.id)}")


# model.eval() # 모델을 평가 모드로 설정
# with torch.no_grad(): # 기울기 계산을 비활성화
#   outputs = model(image_tensor)
#   embeddings = outputs.last_hidden_state.mean(dim=1)


# # Step 5. 추론 결과 후처리(임베딩값)
# embedding_vector = embeddings.numpy().flatten()  # 임베딩 벡터를 numpy 배열로 변환하고 평탄화
# print(f"Embedding Vector : {embedding_vector}")




