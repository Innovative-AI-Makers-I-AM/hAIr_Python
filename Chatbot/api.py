import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel  # 추가
from vector_store import load_vector_store
from llm_setup import setup_llm_and_retrieval_qa

# .env 파일 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class HairStyleRequest(BaseModel):  # 요청 본문을 정의하는 데이터 모델
    message: str

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

@app.post("/hairstyle-recommendations")
async def get_hairstyle_recommendations(request: HairStyleRequest):  # Request 타입 변경
    response_data = await chatbot.run(request.message)
    return {"response": response_data}

@app.on_event("startup")
async def startup_event():
    await chatbot.initialize()  # FastAPI 시작 시 챗봇 초기화

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
