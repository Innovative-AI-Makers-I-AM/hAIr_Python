import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from vector_store import load_vector_store
from llm_setup import setup_llm_and_retrieval_qa

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
        llm_model_name = "gpt-4o"  # "gpt-3.5-turbo" 등으로 변경 가능
        temperature = 0.6
        max_tokens = 300
        self.qa = setup_llm_and_retrieval_qa(db, llm_model_name, temperature, max_tokens, prompt_template)

    
    async def run(self, message):
        response = self.qa({"question": message})
        # response = self.qa(inputs)

        print(response)  # 콘솔에 response 출력
        return response["result"]

chatbot = HairStyleChatbot()  # 챗봇 인스턴스 생성

@app.post("/hairstyle-recommendations")
async def get_hairstyle_recommendations(request: Request):
    print(request)
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
