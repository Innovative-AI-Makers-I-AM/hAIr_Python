# text_face_multimodal.py

## 랜덤 시드 설정 함수
- **목적**: 랜덤 시드를 설정하여 실험의 재현성을 확보
- **인풋**: 시드 값 `42`
- **아웃풋**: 없음
- **코드의 동작에 영향을 미치는 시드 값 설정**
```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
```

## 텍스트 임베딩 모델 로드
- **목적**: 사전 학습된 텍스트 임베딩 모델을 로드하고 평가 모드로 설정
- **인풋**: 사전 학습된 모델의 이름 `upskyy/kf-deberta-multitask`
- **아웃풋**: 텍스트 임베딩 모델과 토크나이저
```python
tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
text_model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask")
text_model.eval()
```

## 얼굴 유형 분류 모델 로드
- **목적**: 사전 학습된 ResNet18 모델을 로드하고 마지막 레이어를 제거하여 512차원 임베딩을 유지
- **인풋**: 사전 학습된 ResNet18 모델
- **아웃풋**: 얼굴 임베딩을 추출하는 모델
```python
face_model = models.resnet18(pretrained=True)
face_model.fc = nn.Identity()  # 마지막 레이어를 제거하여 512차원 출력 유지
face_model.eval()
```

## 텍스트 전처리 함수
- **목적**: 텍스트를 토큰화하고 모델 입력 형식으로 변환
- **인풋**: 원시 텍스트
- **아웃풋**: 토큰화된 텍스트
```python
def preprocess_text(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    return encoded_input
```

## 평균 풀링 함수
- **목적**: 텍스트 임베딩의 평균을 계산하여 문장 임베딩을 생성
- **인풋**: 모델 출력, 어텐션 마스크
- **아웃풋**: 문장 임베딩
```python
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```

## 텍스트 임베딩 추출 함수
- **목적**: 텍스트 임베딩을 추출
- **인풋**: 원시 텍스트
- **아웃풋**: 문장 임베딩
```python
def get_text_embedding(text):
    inputs = preprocess_text(text)
    with torch.no_grad():
        model_output = text_model(**inputs)
    sentence_embeddings = mean_pooling(model_output, inputs['attention_mask'])
    return sentence_embeddings
```

## 이미지 전처리 함수
- **목적**: 이미지를 전처리하여 모델 입력 형식으로 변환
- **인풋**: 이미지 파일 경로
- **아웃풋**: 전처리된 이미지 텐서
```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)
```

## 얼굴 임베딩 추출 함수
- **목적**: 얼굴 임베딩을 추출
- **인풋**: 이미지 파일 경로
- **아웃풋**: 얼굴 임베딩
```python
def get_face_embedding(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        face_embedding = face_model(image)
    return face_embedding
```

## Late Fusion 모델 정의
- **목적**: 텍스트 임베딩과 얼굴 임베딩을 결합하여 최종 분류를 수행
- **인풋**: 텍스트 임베딩, 얼굴 임베딩
- **아웃풋**: 결합된 특성으로부터의 분류 결과
```python
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
```

## 모델 초기화
- **목적**: Late Fusion 모델을 초기화
- **인풋**: 텍스트 임베딩 차원, 얼굴 임베딩 차원, 클래스 수
- **아웃풋**: 초기화된 Late Fusion 모델
```python
fusion_model = LateFusionModel(text_embedding_dim=768, face_embedding_dim=512, num_classes=5)
```

## 예시 예측 과정
- **목적**: 텍스트와 이미지를 입력받아 예측을 수행
- **인풋**: 텍스트, 이미지 파일 경로
- **아웃풋**: 모델의 예측 결과
```python
def predict(text, image_path):
    text_embedding = get_text_embedding(text)
    face_embedding = get_face_embedding(image_path)
    outputs = fusion_model(text_embedding, face_embedding)
    return outputs
```

## 소프트맥스 적용 함수
- **목적**: 로짓을 확률로 변환
- **인풋**: 로짓
- **아웃풋**: 확률
```python
def apply_softmax(logits):
    probabilities = F.softmax(logits, dim=1)
    return probabilities
```

## 예측 클래스 결정 함수
- **목적**: 소프트맥스를 적용하고 예측 클래스를 결정
- **인풋**: 로짓
- **아웃풋**: 예측 클래스, 확률
```python
def predict_class(logits):
    probabilities = apply_softmax(logits)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities
```

## 실행 파일의 디렉토리를 기준으로 이미지 경로 설정
- **목적**: 이미지 파일의 경로를 설정하고 존재 여부를 확인
- **인풋**: 없음
- **아웃풋**: 이미지 파일 경로
```python
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, '..', 'images', 'Autumn Gaze.jpg')

if not os.path.exists(image_path):
    raise FileNotFoundError(f"File not found: {image_path}")
```

## 예시 입력 데이터
- **목적**: 예시 입력 데이터를 설정
- **인풋**: 없음
- **아웃풋**: 텍스트 데이터
```python
text = "안녕하세요? 한국어 문장 임베딩을 위한 모델입니다."
```

## 예측 실행
- **목적**: 텍스트와 이미지를 입력받아 예측을 수행하고 결과를 출력
- **인풋**: 텍스트, 이미지 파일 경로
- **아웃풋**: 예측 클래스, 확률
```python
outputs = predict(text, image_path)
predicted_class, probabilities = predict_class(outputs)
print(f"Predicted class: {predicted_class.item()}")
print(f"Probabilities: {probabilities}")
```