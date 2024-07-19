import os
import random
import numpy as np
import torch

# 랜덤 시드 설정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 시드 값 설정
set_seed(42)

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image

# 텍스트 임베딩 모델 로드
tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
text_model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask")
text_model.eval()

# 얼굴 유형 분류 모델 로드
face_model = models.resnet18(pretrained=True)
face_model.fc = nn.Identity()  # 마지막 레이어를 제거하여 512차원 출력 유지
face_model.eval()

# 텍스트 전처리 함수
def preprocess_text(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    return encoded_input

# 평균 풀링 함수
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 텍스트 임베딩 추출 함수
def get_text_embedding(text):
    if not text:
        return torch.zeros((1, 768))  # 텍스트가 없을 경우 0으로 채운 임베딩 반환
    inputs = preprocess_text(text)
    with torch.no_grad():
        model_output = text_model(**inputs)
    sentence_embeddings = mean_pooling(model_output, inputs['attention_mask'])
    return sentence_embeddings

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# 얼굴 임베딩 추출 함수
def get_face_embedding(image_path):
    if not image_path or not os.path.exists(image_path):
        return torch.zeros((1, 512))  # 이미지가 없거나 경로가 잘못된 경우 0으로 채운 임베딩 반환
    image = preprocess_image(image_path)
    with torch.no_grad():
        face_embedding = face_model(image)
    return face_embedding

# Late Fusion 모델 정의
class LateFusionModel(nn.Module):
    def __init__(self, text_embedding_dim, face_embedding_dim, num_classes):
        super(LateFusionModel, self).__init__()
        self.text_fc = nn.Linear(text_embedding_dim, 128)
        self.face_fc = nn.Linear(face_embedding_dim, 128)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, text_embedding, face_embedding):
        text_feature = self.text_fc(text_embedding)
        face_feature = self.face_fc(face_embedding)
        combined_feature = torch.cat((text_feature, face_feature), dim=1)
        output = self.classifier(combined_feature)
        return output

# 모델 초기화
fusion_model = LateFusionModel(text_embedding_dim=768, face_embedding_dim=512, num_classes=5)

# 예시 예측 과정
def predict(text=None, image_path=None):
    text_embedding = get_text_embedding(text)
    face_embedding = get_face_embedding(image_path)
    if not text and image_path:
        # 텍스트가 없으면 이미지 임베딩만 사용
        combined_embedding = torch.cat((torch.zeros((1, 128)), fusion_model.face_fc(face_embedding)), dim=1)
    elif text and not image_path:
        # 이미지가 없으면 텍스트 임베딩만 사용
        combined_embedding = torch.cat((fusion_model.text_fc(text_embedding), torch.zeros((1, 128))), dim=1)
    else:
        # 둘 다 있으면 둘 다 사용
        combined_embedding = torch.cat((fusion_model.text_fc(text_embedding), fusion_model.face_fc(face_embedding)), dim=1)
    outputs = fusion_model.classifier(combined_embedding)
    return outputs, text_embedding, face_embedding, combined_embedding

# 소프트맥스 적용 함수
def apply_softmax(logits):
    probabilities = F.softmax(logits, dim=1)
    return probabilities

# 예측 클래스 결정 함수
def predict_class(logits):
    probabilities = apply_softmax(logits)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities

# 실행 파일의 디렉토리를 기준으로 이미지 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, '..', 'images', 'KakaoTalk_20240709_134904035.jpg')

# 파일이 존재하는지 확인
if not os.path.exists(image_path):
    raise FileNotFoundError(f"File not found: {image_path}")

# 예시 입력 데이터
text = "안녕하세요? 한국어 문장 임베딩을 위한 모델입니다."
# text = ""
# image_path = "";

# 예측 실행 (텍스트와 이미지 둘 다 입력)
outputs, text_embedding, face_embedding, combined_embedding = predict(text, image_path)
predicted_class, probabilities = predict_class(outputs)
print(f"Predicted class (both inputs): {predicted_class.item()}")
print(f"Probabilities: {probabilities}")
print(f"Text Embedding: {text_embedding}")
print(f"Face Embedding: {face_embedding}")
print(f"Combined Embedding: {combined_embedding}")

# 예측 실행 (텍스트만 입력)
outputs, text_embedding, face_embedding, combined_embedding = predict(text=text)
predicted_class, probabilities = predict_class(outputs)
print(f"Predicted class (text only): {predicted_class.item()}")
print(f"Probabilities: {probabilities}")
print(f"Text Embedding: {text_embedding}")
print(f"Face Embedding: {face_embedding}")
print(f"Combined Embedding: {combined_embedding}")

# 예측 실행 (이미지만 입력)
outputs, text_embedding, face_embedding, combined_embedding = predict(image_path=image_path)
predicted_class, probabilities = predict_class(outputs)
print(f"Predicted class (image only): {predicted_class.item()}")
print(f"Probabilities: {probabilities}")
print(f"Text Embedding: {text_embedding}")
print(f"Face Embedding: {face_embedding}")
print(f"Combined Embedding: {combined_embedding}")
