import os
#추가
from batch_processor import batch_process_documents

def main():
    persist_directory = "./chroma_db"
    collection_metadata = {'hnsw:space': 'cosine'}
    model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        data_dir = "crawled_data"
        batch_size = 10  # 한번에 처리할 파일의 수
        batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata)
        print("Batch processing complete. All documents processed and stored.")
    else:
        print("Database already exists.")

if __name__ == "__main__":
    main()


#------------------------------------------------필요 없는 코드--------------------------------------------------------------


# # .env 파일 로드
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# class HairStyleChatbot:
#     def __init__(self):
#         self.qa = None

#     async def initialize(self):
#         persist_directory = "./chroma_db"
#         collection_metadata = {'hnsw:space': 'cosine'}
#         model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

        
#         if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
#             data_dir = "crawled_data"
#             #추가된 부분
#             batch_size = 11  # 한번에 처리할 파일의 수
#             batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata)
#             db = load_vector_store(persist_directory, model_name)  # 생성된 DB 로드
#         else:
#             db = load_vector_store(persist_directory, model_name)


#         #     docs = load_and_split_documents(data_dir)
#         #     db = create_vector_store(docs, model_name, persist_directory, collection_metadata)
#         # else:
#         #     db = load_vector_store(persist_directory, model_name)

#         prompt_template = (
#             "너는 능력 있는 헤어디자이너로, 유저가 원하는 헤어스타일을 추천하는 역할을 맡고 있어. "
#             "처음 대화할 때만 인사하고, 이후에는 인사를 생략해. "
#             "자연스럽게 대화를 이어가면서 유저의 답변을 이끌어내. "
#             "답변할 때는 항상 친절하고 공손하게 말해줘. "
#             "답변에 필요한 정보는 retriever 검색 결과를 우선 사용하고 추가적인 정보를 검색해서 정리한 후 답변을 해줘. "
#             "모르는 질문이 나오면 솔직하게 '모르겠다'고 말해줘."
#         )
#         llm_model_name = "gpt-4o"  # "gpt-3.5-turbo" 등으로 변경 가능
#         temperature = 0.8
#         max_tokens = 1024
#         self.qa = setup_llm_and_retrieval_qa(db, llm_model_name, temperature, max_tokens, prompt_template)


#     async def run(self, message):
#         response = self.qa({"query": message})
#         print(response)  # 콘솔에 response 출력
#         return response["result"]
#         # response = await self.qa({"query": message})  # 수정된 부분: await 추가
#         # return {
#         #     "result": response["result"],
#         #     "sources": response.get("sources", [])
#         # }

# chatbot = HairStyleChatbot()  # 챗봇 인스턴스 생성

# @app.post("/hairstyle-recommendations")
# async def get_hairstyle_recommendations(request: Request):
#     request_data = await request.json()
#     message = request_data.get("message")
#     response_data = await chatbot.run(message)
#     return {"response": response_data}

# @app.on_event("startup")
# async def startup_event():
#     await chatbot.initialize()  # FastAPI 시작 시 챗봇 초기화

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request
# from data_processing import load_and_split_documents
# from vector_store import create_vector_store, load_vector_store
# from llm_setup import setup_llm_and_retrieval_qa

# class HairStyleChatbot:
#     def __init__(self):
#         self.qa = None

#     async def initialize(self):
#         persist_directory = "./chroma_db"
#         collection_metadata = {'hnsw:space': 'cosine'}
#         model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

#         if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
#             data_dir = "crawled_data"
#             docs = load_and_split_documents(data_dir)
#             db = create_vector_store(docs, model_name, persist_directory, collection_metadata)
#         else:
#             db = load_vector_store(persist_directory, model_name)

#         prompt_template = (
#             "너의 역할은 실력이 뛰어난 헤어디자이너인데 유저가 원하는 헤어스타일을 추천해주는 걸 주 업무로 하고 있어. "
#             "대화를 시작하면 처음 한번만 인사를 하고 이후로는 인사하지 마. "
#             "그리고 유저가 너에게 헤어 스타일과 관련된 질문을 하면 헤어스타일에 대한 설명을 해줘. "
#             "예를 들면 헤어스타일을 추천해줘 이런 질문을 한다면 어떤 스타일을 원하세요? 라고 물어보고 답변을 듣고 적절한 걸 추천해줘. "
#             "답변에는 언제나 친절하게 높임말을 사용해서 대답을 해야해. "
#             "유저와 자연스러운 대화를 이끌어가면서 유저의 답변을 유도하는 식으로 대화를 진행해. "
#             "유저의 질문에 대한 헤어스타일 답변에 대한 정보는 retriever 검색 정보와 함께 부족한 부분은 추가적인 정보들을 검색해서 포함하고 가장 적절한 답변을 깔끔하게 요약해서 답변을 해줘. "
#             "만약 답변을 모르는 경우 솔직하게 '모르겠다'고 말해줘."
#         )
#         llm_model_name = "gpt-4o"  # "gpt-3.5-turbo" 등으로 변경 가능
#         temperature = 0.8
#         max_tokens = 1024
#         self.qa = setup_llm_and_retrieval_qa(db, llm_model_name, temperature, max_tokens, prompt_template)

#     async def run(self, message):
#         response = self.qa({"query": message})
#         return response["result"]

# chatbot = HairStyleChatbot()  # 챗봇 인스턴스 생성

# @app.post("/hairstyle-recommendations")
# async def get_hairstyle_recommendations(request: Request):
#     request_data = await request.json()
#     message = request_data.get("message")
#     response_data = await chatbot.run(message)
#     return {"response": response_data}

# @app.on_event("startup")
# async def startup_event():
#     await chatbot.initialize()  # FastAPI 시작 시 챗봇 초기화

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
