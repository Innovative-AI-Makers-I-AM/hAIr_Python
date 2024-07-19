import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 얼굴 유형 분류 모델 로드
face_model = models.resnet18(pretrained=True)
face_model.fc = nn.Linear(face_model.fc.in_features, 5)  # 5개의 얼굴 유형 분류
face_model.load_state_dict(torch.load('path_to_saved_face_model.pth'))
face_model.eval()

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
    image = preprocess_image(image_path)
    with torch.no_grad():
        face_embedding = face_model(image)
    return face_embedding

# 예시 이미지 경로
image_path = 'images\Autumn Gaze.jpg'
embedding = get_face_embedding(image_path)
print(embedding)
