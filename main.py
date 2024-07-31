import os
import base64
from io import BytesIO
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from matplotlib import font_manager
import matplotlib
from gradio_client import Client, handle_file
from Chatbot.vector_store import load_vector_store
from Chatbot.llm_setup import setup_llm_and_retrieval_qa
from Chatbot.chroma_db_utils import initialize_chroma_db, search_chroma_with_filter
import tempfile
import io

# .env 파일 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용, 필요에 따라 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace API 클라이언트 초기화
client = Client("AIRI-Institute/HairFastGAN")

# 이미지 프로세서 및 모델 초기화
processor = AutoImageProcessor.from_pretrained("metadome/face_shape_classification")
model = AutoModelForImageClassification.from_pretrained("metadome/face_shape_classification")
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
            """
                You are a professional hairstylist chatbot. 
                Recommend hairstyles that match the user's preferences. 
                Greet the user only in the first conversation and omit it thereafter. 
                Communicate naturally and concisely. Respond politely and courteously. 
                
                For questions you don't know, respond with "I don't know." Always respond in Korean, 
                summarizing your answers within 2-3 sentences. 
                Ensure that your responses are clean and do not include numbers, bold text, or special characters.
                Remember previous conversations and use that context to answer questions consistently.
                
                Do not include unnecessary information that does not align with the user's question. Provide only accurate and relevant content.
                If the user asks for a summary, provide a summary based solely on the conversation without adding any additional information
            """
        )
        llm_model_name = "gpt-4o"
        temperature = 0.8
        max_tokens = 200
        self.qa = setup_llm_and_retrieval_qa(db, llm_model_name, temperature, max_tokens, prompt_template)

    async def run(self, message: str):  # message 타입 명시
        if not self.qa:
            raise HTTPException(status_code=500, detail="Chatbot not initialized.")
        response = self.qa({"question": message})
        return response["result"]

chatbot = HairStyleChatbot()

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
        search_embedding = generate_image_embedding_with_base_64(request.image_base64)
        filter_conditions = {
            'sex': request.sex,
            'length': request.length,
            'style': request.styles
        }
        results = search_all_collections_with_filter(chroma_client, "image_collection", search_embedding, filter_conditions, k=request.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching hairstyles: {e}")


def generate_image_embedding(image_data: bytes):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")  # 이미지 읽기 및 RGB 형식으로 변환
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    image_embedding = outputs[0].numpy().flatten().tolist()
    return image_embedding

@app.post("/search-hairstyles")
async def search_hairstyles(
    file: UploadFile = File(...),
    sex: str = Form(...),
    length: str = Form(...),
    styles: str = Form(...),
    k: int = 5
):
    try:
        print(f"Received file: {file.filename}")
        print(f"Received sex: {sex}")
        print(f"Received length: {length}")
        print(f"Received styles: {styles}")

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
        # print(encoded_images)
        return {"results": encoded_images}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching hairstyles: {e}")


# HairFastGAN 관련 엔드포인트
@app.get("/")
async def read_root():
    return {"message": "Welcome to the HairFastGAN API. Use the /hair_transfer endpoint to upload images."}

# @app.post("/hair_transfer")
# async def hair_transfer(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
#     # Read the uploaded images into memory
#     face_image = Image.open(io.BytesIO(await face.read()))
#     shape_image = Image.open(io.BytesIO(await shape.read()))
#     color_image = Image.open(io.BytesIO(await color.read()))

#     # Save the images to temporary files
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as face_tempfile:
#         face_image.save(face_tempfile, format="JPEG")
#         face_tempfile_path = face_tempfile.name
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as shape_tempfile:
#         shape_image.save(shape_tempfile, format="PNG")
#         shape_tempfile_path = shape_tempfile.name
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as color_tempfile:
#         color_image.save(color_tempfile, format="PNG")
#         color_tempfile_path = color_tempfile.name

#     result_image_path = None  # 초기화

#     try:
#         # Step 1: Resize the face image
#         resized_face = client.predict(
#             img=handle_file(face_tempfile_path),
#             align=["Face"],
#             api_name="/resize_inner"
#         )

#         # Step 2: Resize the hairstyle image
#         resized_shape = client.predict(
#             img=handle_file(shape_tempfile_path),
#             align=["Shape"],
#             api_name="/resize_inner_1"
#         )

#         # Step 3: Resize the hair color image
#         resized_color = client.predict(
#             img=handle_file(color_tempfile_path),
#             align=["Color"],
#             api_name="/resize_inner_2"
#         )

#         # Extracting the resized image paths
#         resized_face_path = resized_face if isinstance(resized_face, str) else resized_face[0]
#         resized_shape_path = resized_shape if isinstance(resized_shape, str) else resized_shape[0]
#         resized_color_path = resized_color if isinstance(resized_color, str) else resized_color[0]

#         # Read the resized images into memory
#         resized_face_image = Image.open(resized_face_path)
#         resized_shape_image = Image.open(resized_shape_path)
#         resized_color_image = Image.open(resized_color_path)

#         resized_face_bytes = io.BytesIO()
#         resized_shape_bytes = io.BytesIO()
#         resized_color_bytes = io.BytesIO()

#         resized_face_image.save(resized_face_bytes, format="PNG")
#         resized_shape_image.save(resized_shape_bytes, format="PNG")
#         resized_color_image.save(resized_color_bytes, format="PNG")

#         resized_face_bytes.seek(0)
#         resized_shape_bytes.seek(0)
#         resized_color_bytes.seek(0)

#         # Step 4: Swap hair using resized images
#         result = client.predict(
#             face=handle_file(resized_face_path),
#             shape=handle_file(resized_shape_path),
#             color=handle_file(resized_color_path),
#             blending="Article",
#             poisson_iters=0,
#             poisson_erosion=15,
#             api_name="/swap_hair"
#         )

#         # Extract the result image path
#         result_image_path = result[0]['value']

#         # Read the result image
#         result_image = Image.open(result_image_path)
#         result_image_bytes = io.BytesIO()
#         result_image.save(result_image_bytes, format="PNG")
#         result_image_bytes.seek(0)

#         return StreamingResponse(result_image_bytes, media_type="image/png")

#     finally:
#         # Clean up temporary files
#         os.remove(face_tempfile_path)
#         os.remove(shape_tempfile_path)
#         os.remove(color_tempfile_path)
#         if resized_face_path and os.path.exists(resized_face_path):
#             os.remove(resized_face_path)
#         if resized_shape_path and os.path.exists(resized_shape_path):
#             os.remove(resized_shape_path)
#         if resized_color_path and os.path.exists(resized_color_path):
#             os.remove(resized_color_path)
#         if result_image_path and os.path.exists(result_image_path):
#             os.remove(result_image_path)

@app.on_event("startup")
async def startup_event():
    await chatbot.initialize()  # FastAPI 시작 시 챗봇 초기화

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
