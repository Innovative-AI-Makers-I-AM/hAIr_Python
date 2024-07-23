import os
from dotenv import load_dotenv
from data_processing import load_and_split_documents
from vector_store import create_vector_store, load_vector_store
from llm_setup import setup_llm_and_retrieval_qa
from chat import run_chatbot

# .env 파일 로드
load_dotenv()

# 환경 변수에서 OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def main():
    # 벡터 데이터베이스 디렉토리 설정
    persist_directory = "./chroma_db"
    collection_metadata = {'hnsw:space': 'cosine'}
    model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        # 데이터 로드 및 전처리
        data_dir = "crawled_data"
        docs = load_and_split_documents(data_dir)

        # 벡터 임베딩 및 데이터베이스 생성
        db = create_vector_store(docs, model_name, persist_directory, collection_metadata)
    else:
        # 벡터 데이터베이스 로드
        db = load_vector_store(persist_directory, model_name)

    # LLM 및 RAG 설정
    prompt_template = (
        "너는 친절하고 유머 감각이 뛰어난 헤어봇이야 "
        "유저가 너에게 헤어 스타일과 관련된 질문을 하면 헤어스타일에 대한 설명을 해줘."
        "답변에는 언제나 친절하게 높임말을 사용해서 대답을 해야해. "
        "retriever 검색 정보를 사용하고 추가로 정보를 검색하여 답변에 포함시켜줘. "
        "만약 답변을 모르는 경우 솔직하게 '모르겠다'고 말해줘."
        )
    llm_model_name = "gpt-4o"
    temperature = 0.8
    max_tokens = 1024
    qa = setup_llm_and_retrieval_qa(db, llm_model_name, temperature, max_tokens, prompt_template)

    # 챗봇 실행
    run_chatbot(qa)

if __name__ == "__main__":
    main()
