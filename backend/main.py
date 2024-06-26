from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI  # 올바른 경로로 가져오기

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:3000",  # React 개발 서버
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API 키 설정
openai_api_key = ""

# 초기 프롬프트 템플릿 설정 (한국어)
initial_prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "당신은 전문 헤어 디자이너입니다. 사용자가 미용실에 있는 것처럼 대화를 나누세요. "
        "사용자의 질문에만 답변하고, 전문 헤어 디자이너처럼 대화 해주세요. "
        "사용자: {input_text}\n"
        "전문 헤어 디자이너:"
    )
)

# 대화 지속을 위한 프롬프트 템플릿
ongoing_prompt_template = PromptTemplate(
    input_variables=["conversation_history", "input_text"],
    template=(
        "당신은 전문 헤어 디자이너입니다. 다음은 사용자와의 대화 내용입니다:\n"
        "{conversation_history}\n"
        "사용자: {input_text}\n"
        "전문 헤어 디자이너:"
    )
)

# 요약을 위한 프롬프트 템플릿
summary_prompt_template = PromptTemplate(
    input_variables=["conversation_history"],
    template=(
        "당신은 전문 헤어 디자이너입니다. 다음은 사용자와의 대화 내용입니다:\n"
        "{conversation_history}\n"
        "사용자가 원하는 머리스타일을 간단히 요약해 주세요."
    )
)

# LLMChain 설정
llm_chain = LLMChain(
    llm=OpenAI(api_key=openai_api_key, max_tokens=150, temperature=0.7, top_p=0.9),
    prompt=initial_prompt_template
)

conversation_history = []

# 데이터 모델 정의
class ChatMessage(BaseModel):
    message: str

# Static 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=JSONResponse)
async def home(request: Request):
    return {"message": "Welcome to the hair designer chat API!"}

@app.post("/chat")
async def chat(request: ChatMessage):
    user_input = request.message
    conversation_history.append(f"사용자: {user_input}")
    
    # 대화 이력을 문자열로 결합
    conversation_text = "\n".join(conversation_history)
    
    # LLMChain 실행
    ongoing_chain = LLMChain(
        llm=OpenAI(api_key=openai_api_key, max_tokens=150, temperature=0.7, top_p=0.9),
        prompt=ongoing_prompt_template
    )
    response = ongoing_chain.run(conversation_history=conversation_text, input_text=user_input)
    
    conversation_history.append(f"전문 헤어 디자이너: {response.strip()}")

    return JSONResponse(content={"response": response.strip()})

@app.post("/end_chat")
async def end_chat():
    # 대화 이력을 문자열로 결합
    conversation_text = "\n".join(conversation_history)
    
    # 요약 프롬프트 실행
    summary_chain = LLMChain(
        llm=OpenAI(api_key=openai_api_key, max_tokens=150, temperature=0.7, top_p=0.9),
        prompt=summary_prompt_template
    )
    summary_response = summary_chain.run(conversation_history=conversation_text)
    
    # 대화 이력을 초기화
    conversation_history.clear()
    
    return JSONResponse(content={"response": summary_response.strip()})

@app.get("/init_chat")
async def init_chat():
    initial_message = "안녕하세요! 어떤 머리스타일을 원하시나요?"
    conversation_history.append(f"전문 헤어 디자이너: {initial_message}")
    return JSONResponse(content={"response": initial_message})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)