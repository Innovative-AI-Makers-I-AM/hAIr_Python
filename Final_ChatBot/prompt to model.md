api.py
```
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
        model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        db = load_vector_store(persist_directory, model_name)

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
```

batch_processor.py
```
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, load_vector_store

def batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata):
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    total_files = len(files)
    for i in range(0, total_files, batch_size):
        batch_files = files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        documents = []
        for file in batch_files:
            print(f"Loading file: {file}")
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")
            loaded_docs = loader.load_and_split()
            print(f"Loaded {len(loaded_docs)} documents from {file}")
            documents.extend(loaded_docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function=len)
        split_docs = text_splitter.split_documents(documents)
        print(f"Total number of split documents: {len(split_docs)}")
        
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            db = create_vector_store(split_docs, model_name, persist_directory, collection_metadata)
        else:
            db = load_vector_store(persist_directory, model_name)
            db.add_documents(split_docs)
```

chat.py
```
def run_chatbot(qa):
    while True:
        query = input("질문: ")
        if query in ["exit", "quit"]:
            break
        for token in qa.stream({"query": query}):
            print(token['result'], end='', flush=True)
        print()
```

data_processing.py
```
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            print(f"Loading file: {file}")  # 파일 로딩 확인
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")
            loaded_docs = loader.load_and_split()
            print(f"Loaded {len(loaded_docs)} documents from {file}")  # 로드한 문서 개수 확인
            documents.extend(loaded_docs)
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function = len)
    split_docs = text_splitter.split_documents(documents)
    print(f"Total number of split documents: {len(split_docs)}")  # 스플릿한 문서 개수 확인


    return split_docs
```

llm_setup.py
```
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            print(f"Loading file: {file}")  # 파일 로딩 확인
            loader = TextLoader(f"{data_dir}/{file}", encoding="utf-8")
            loaded_docs = loader.load_and_split()
            print(f"Loaded {len(loaded_docs)} documents from {file}")  # 로드한 문서 개수 확인
            documents.extend(loaded_docs)
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function = len)
    split_docs = text_splitter.split_documents(documents)
    print(f"Total number of split documents: {len(split_docs)}")  # 스플릿한 문서 개수 확인


    return split_docs
```

main.py
```
import os
from batch_processor import batch_process_documents

def main():
    # 
    persist_directory = "./chroma_db"
    collection_metadata = {'hnsw:space': 'cosine'}
    model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        data_dir = "crawled_data"
        batch_size = 11  # 한번에 처리할 파일의 수
        batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata)
        print("Batch processing complete. All documents processed and stored.")
    else:
        print("Database already exists.")

if __name__ == "__main__":
    main()
```

vector_store.py
```
import os  # os 모듈 임포트
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(docs, model_name, persist_directory, collection_metadata):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory, collection_metadata=collection_metadata)

    # 문서 개수 출력
    print(f"Number of documents create stored: {db._collection.count()}")

    return db

def load_vector_store(persist_directory, model_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 문서 개수 출력
    print(f"Number of documents load stored: {db._collection.count()}")
    
    return db
```
해당 코드들을 이해하고 유치원생들도 이해할 수 있도록 한국어로 설명해줘