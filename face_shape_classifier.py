import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
# import requests

# 모델 및 프로세서 로드
processor = AutoImageProcessor.from_pretrained("metadome/face_shape_classification")
model = AutoModelForImageClassification.from_pretrained("metadome/face_shape_classification")

# 이미지 로드 함수
def load_image(image_path):
    return Image.open(image_path)

# 예측 함수
# def predict(image_path):
#     image = load_image(image_path)
#     inputs = processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
#     return model.config.id2label[predicted_class_idx]

# 이미지 경로 없는 예측 함수 
def predict(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# 예측 수행

# image_path = "images\KakaoTalk_20240709_134904035.jpg"  # 예측할 이미지 경로
# predicted_label = predict(image_path)
# print(f"Predicted label: {predicted_label}")
