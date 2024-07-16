from transformers import CLIPProcessor, CLIPModel
import torch
import pinecone
import os
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# .env 파일 로드
load_dotenv()

# Pinecone API 키 및 환경 변수 설정
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Pinecone 초기화
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# 사용자 입력 예시
user_input = "긴 머리"

# 텍스트 임베딩 생성 함수
def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features

# 텍스트 임베딩 생성
text_embedding = get_text_embedding(user_input)
text_vector = text_embedding.cpu().numpy().flatten()

# 벡터 정규화 및 NaN 값 처리
text_vector = np.nan_to_num(text_vector)  # NaN, -inf, inf 값을 0으로 변환
norm = np.linalg.norm(text_vector)
if norm != 0:
    text_vector = text_vector / norm

# 벡터 값의 기본 통계 확인
min_value = np.min(text_vector)
max_value = np.max(text_vector)
mean_value = np.mean(text_vector)
std_value = np.std(text_vector)

print(f"Min value: {min_value}")
print(f"Max value: {max_value}")
print(f"Mean value: {mean_value}")
print(f"Standard deviation: {std_value}")

# 벡터 값의 분포 시각화
plt.hist(text_vector, bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Text Vector Values')
plt.show()

# 벡터 값을 -1과 1 사이로 정규화
text_vector = np.clip(text_vector, -1, 1)

# 벡터 값을 float32 타입으로 변환
text_vector = text_vector.astype(np.float32)

# 텍스트 벡터의 유효성 점검 (모든 값이 유효한 숫자인지 확인)
if not np.isfinite(text_vector).all():
    raise ValueError("Text vector contains invalid values.")

# 텍스트 임베딩 벡터의 크기 확인
print(f"Text embedding vector size: {text_vector.size}")

# Pinecone 인덱스의 벡터 크기 확인
index_info = index.describe_index_stats()
print(f"Pinecone index dimension: {index_info['dimension']}")

# 벡터 값 출력 (추가)
print(f"Text vector values: {text_vector}")

# 유사 이미지 검색
result = index.query(queries=[text_vector.tolist()], top_k=5)

# 검색된 이미지 경로
image_indices = [match['id'] for match in result['matches']]
similar_image_paths = [index.fetch([id])['vectors'][id]['metadata']['path'] for id in image_indices]

# 검색된 이미지 경로 출력 및 표시
print(similar_image_paths)
for image_path in similar_image_paths:
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()