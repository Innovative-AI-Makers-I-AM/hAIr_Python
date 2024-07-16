from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from torchvision import transforms
import pinecone
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Pinecone API 키 및 환경 변수 설정
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Pinecone 초기화
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# 인덱스 생성
# if pinecone_index_name not in pc.list_indexes():
#     pc.create_index(
#         name=pinecone_index_name,
#         dimension=512,  # CLIP 모델의 임베딩 차원에 맞게 설정
#         metric='cosine',  # 유사도 메트릭 설정
#         spec=pinecone.ServerlessSpec(
#             cloud='aws',  # 사용할 클라우드 제공자
#             region='us-east-1'  # 사용할 지역
#         )
#     )

# 인덱스 연결
index = pc.Index(pinecone_index_name)

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# 이미지 경로 리스트 생성 함수
def get_image_paths(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png')  # 처리할 이미지 파일 확장자
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)
                   if filename.lower().endswith(valid_extensions)]
    return image_paths

# 이미지 경로 리스트
image_directory = "images"
image_paths = get_image_paths(image_directory)

# 이미지 임베딩 생성 함수
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

# 모든 이미지 임베딩 생성 및 Pinecone에 업로드
for i, image_path in enumerate(image_paths):
    image_embedding = get_image_embedding(image_path)
    vector = image_embedding.cpu().numpy().flatten().tolist()
    metadata = {"path": image_path, "description": f"Image {i}"}
    index.upsert(vectors=[(str(i), vector, metadata)])
