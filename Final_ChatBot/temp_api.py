import os
import base64
from io import BytesIO
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel  # 추가
from vector_store import load_vector_store
from llm_setup import setup_llm_and_retrieval_qa
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from matplotlib import font_manager
import matplotlib
from chroma_db_utils import initialize_chroma_db, search_chroma_with_filter

# .env 파일 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 이미지 프로세서 및 모델 초기화
processor = AutoImageProcessor.from_pretrained("metadome/face_shape_classification")
model = AutoModelForImageClassification.from_pretrained("metadome/face_shape_classification")

# 마지막 레이어 제거 (모델의 특징 추출 부분 사용)
model.classifier = torch.nn.Identity()
model.eval()

# 폰트 설정 (한글 폰트 경로 설정)
if os.name == 'nt':  # 윈도우
    font_path = "C:/Windows/Fonts/malgun.ttf"
elif os.name == 'posix':  # 맥OS 및 리눅스
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 경로는 시스템에 따라 다를 수 있음
else:
    font_path = None

if font_path and os.path.exists(font_path):
    font = font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rc('font', family=font)
else:
    print("Font file not found. Please check the font path.")

app = FastAPI()

class HairStyleRequest(BaseModel):  # 요청 본문을 정의하는 데이터 모델
    message: str

class SearchRequest(BaseModel):
    image_base64: str
    sex: Optional[str] = None
    length: Optional[str] = None
    styles: Optional[str] = None
    k: int = 5

class ImageResponse(BaseModel):
    image: str  # Base64 인코딩된 이미지 데이터

# Chroma DB 초기화
persist_directory = "./chroma_image_db"
chroma_client = initialize_chroma_db(persist_directory)

class HairStyleChatbot:
    def __init__(self):
        self.qa = None

    async def initialize(self):
        persist_directory = "./chroma_db"
        model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        try:
            db = load_vector_store(persist_directory, model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load vector store: {e}")

        prompt_template = (
            "너는 능력 있는 헤어디자이너로, 유저가 원하는 헤어스타일을 추천하는 역할을 맡고 있어. "
            "처음 대화할 때만 인사하고, 이후에는 인사를 생략해. "
            "자연스럽게 대화를 이어가면서 유저의 답변을 이끌어내. "
            "답변할 때는 항상 친절하고 공손하게 말해줘. "
            "답변에 필요한 정보는 retriever 검색 결과를 우선 사용하고 추가적인 정보를 검색해서 정리한 후 답변을 해줘. "
            "모르는 질문이 나오면 솔직하게 '모르겠다'고 말해줘."
        )
        llm_model_name = "gpt-4o"
        temperature = 0.8
        max_tokens = 1024
        self.qa = setup_llm_and_retrieval_qa(db, llm_model_name, temperature, max_tokens, prompt_template)

    async def run(self, message: str):  # message 타입 명시
        if not self.qa:
            raise HTTPException(status_code=500, detail="Chatbot not initialized.")
        response = self.qa({"query": message})
        return response["result"]

chatbot = HairStyleChatbot()

def generate_image_embedding(image_data: bytes):
    image = Image.open(BytesIO(image_data))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    image_embedding = outputs[0].numpy().flatten().tolist()
    return image_embedding

def generate_image_embedding_with_base_64(image_base64: str):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    image_embedding = outputs[0].numpy().flatten().tolist()
    return image_embedding

def search_all_collections_with_filter(chroma_client, collection_prefix, query_embedding, filter_conditions, k=5):
    collection_index = 0
    all_results = []
    while True:
        collection_name = f"{collection_prefix}{collection_index}"
        try:
            results = search_chroma_with_filter(chroma_client, collection_name, query_embedding, filter_conditions, k)
            all_results.extend(zip(results['distances'][0], results['metadatas'][0]))
            collection_index += 1
        except Exception:
            break
    all_results.sort(key=lambda x: x[0])  # 거리 기준으로 정렬
    top_k_results = all_results[:k]
    return top_k_results

def encode_images_to_base64(image_paths: List[str]) -> List[ImageResponse]:
    encoded_images = []
    for path in image_paths:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append(ImageResponse(image=encoded_string))
    return encoded_images

@app.post("/hairstyle-recommendations")
async def get_hairstyle_recommendations(request: HairStyleRequest):  # Request 타입 변경
    response_data = await chatbot.run(request.message)
    return {"response": response_data}

@app.post("/search-hairstyles-with-base64")
async def search_hairstyles(request : SearchRequest):
    try:
        search_embedding = generate_image_embedding_with_base_64(request.image_path)
        filter_conditions = {
            'sex': request.sex,
            'length': request.length,
            'style': request.style
        }
        results = search_all_collections_with_filter(chroma_client, "image_collection", search_embedding, filter_conditions, k=request.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching hairstyles: {e}")

# @app.post("/search-hairstyles")
# async def search_hairstyles(file: UploadFile = File(...), sex: Optional[str] = None, length: Optional[str] = None, styles: Optional[str] = None, k: int = 5):
#     try:
#         # 이미지 파일을 읽고 Base64 인코딩
#         image_data = await file.read()
#         search_embedding = generate_image_embedding(image_data)
#         filter_conditions = {}
#         if sex:
#             filter_conditions['sex'] = sex
#         if length:
#             filter_conditions['length'] = length
#         if styles:
#             filter_conditions['style'] = styles

#         results = search_all_collections_with_filter(chroma_client, "image_collection", search_embedding, filter_conditions, k=k)
#         return {"results": results}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error searching hairstyles: {e}")

@app.post("/search-hairstyles")
async def search_hairstyles(file: UploadFile = File(...), sex: Optional[str] = None, length: Optional[str] = None, styles: Optional[str] = None, k: int = 5):
    try:
        # 이미지 파일을 읽고 Base64 인코딩
        image_data = await file.read()
        search_embedding = generate_image_embedding(image_data)
        filter_conditions = {}
        if sex:
            filter_conditions['sex'] = sex
        if length:
            filter_conditions['length'] = length
        if styles:
            filter_conditions['style'] = styles

        results = search_all_collections_with_filter(chroma_client, "image_collection", search_embedding, filter_conditions, k=k)
        
        # 예제 파일 경로, 실제 파일 경로로 변경 필요
        image_paths = [result[1]['image_path'] for result in results]
        encoded_images = encode_images_to_base64(image_paths)
        
        return {"results": encoded_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching hairstyles: {e}")

@app.on_event("startup")
async def startup_event():
    await chatbot.initialize()  # FastAPI 시작 시 챗봇 초기화

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)