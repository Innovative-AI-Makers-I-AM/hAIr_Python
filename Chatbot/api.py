import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from vector_store import load_vector_store
from llm_setup import setup_llm_and_retrieval_qa

#추가
from batch_processor import batch_process_documents

# .env 파일 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class HairStyleChatbot:
    def __init__(self):
        self.qa = None

    async def initialize(self):
        persist_directory = "./chroma_db"
        # collection_metadata = {'hnsw:space': 'cosine'}
        model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        db = load_vector_store(persist_directory, model_name)
        
        #------------ 이부분은 필요 없는 코드
        # if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        #     data_dir = "crawled_data"
        #     #추가된 부분
        #     batch_size = 11  # 한번에 처리할 파일의 수
        #     batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata)
        #     db = load_vector_store(persist_directory, model_name)  # 생성된 DB 로드
        # else:
        #     db = load_vector_store(persist_directory, model_name)


        #     docs = load_and_split_documents(data_dir)
        #     db = create_vector_store(docs, model_name, persist_directory, collection_metadata)
        # else:
        #     db = load_vector_store(persist_directory, model_name)

        prompt_template = (
            "너는 능력 있는 헤어디자이너로, 유저가 원하는 헤어스타일을 추천하는 역할을 맡고 있어. "
            "처음 대화할 때만 인사하고, 이후에는 인사를 생략해. "
            "자연스럽게 대화를 이어가면서 유저의 답변을 이끌어내. "
            "답변할 때는 항상 친절하고 공손하게 말해줘. "
            "답변에 필요한 정보는 retriever 검색 결과를 우선 사용하고 추가적인 정보를 검색해서 정리한 후 답변을 해줘. "
            "모르는 질문이 나오면 솔직하게 '모르겠다'고 말해줘."
        )
        llm_model_name = "gpt-4o"  # "gpt-3.5-turbo" 등으로 변경 가능
        temperature = 0.8
        max_tokens = 1024
        self.qa = setup_llm_and_retrieval_qa(db, llm_model_name, temperature, max_tokens, prompt_template)


    async def run(self, message):
        response = self.qa({"query": message})
        print(response)  # 콘솔에 response 출력
        return response["result"]
        # response = await self.qa({"query": message})  # 수정된 부분: await 추가
        # return {
        #     "result": response["result"],
        #     "sources": response.get("sources", [])
        # }

chatbot = HairStyleChatbot()  # 챗봇 인스턴스 생성

@app.post("/hairstyle-recommendations")
async def get_hairstyle_recommendations(request: Request):
    request_data = await request.json()
    message = request_data.get("message")
    response_data = await chatbot.run(message)
    return {"response": response_data}

@app.on_event("startup")
async def startup_event():
    await chatbot.initialize()  # FastAPI 시작 시 챗봇 초기화

if __name__ == "__api__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)