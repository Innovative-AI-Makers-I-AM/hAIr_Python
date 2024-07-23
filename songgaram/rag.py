import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# .env 파일 로드 (API 키를 .env 파일에 저장하는 것을 권장합니다)
load_dotenv()

# OpenAI API 키 설정 (환경 변수에서 가져오기)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # .env 파일의 OPENAI_API_KEY 값을 환경 변수에 설정

# 1단계: TXT 문서 로드 및 분할
documents = []
for file in os.listdir("docs"):
    if file.endswith(".txt"):
        loader = TextLoader(f"docs/{file}", encoding="utf-8")  # 인코딩 명시
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 2단계: 임베딩 및 벡터 저장소 생성
embeddings = HuggingFaceInstructEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db", collection_name="hair_description")
db.persist()

# 3단계: LLM 모델 로드 (ChatGPT 4 사용)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1, max_tokens=1024)  # 환경 변수에서 API 키를 자동으로 가져옴

# 4단계: RetrievalQA 체인 생성
# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# 5단계: 질문 입력 및 답변 출력 (반복)
while True:
    query = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
    if query.lower() == "exit":
        break

    # 질문에 대한 답변 및 출처 문서 가져오기
    result = qa_chain.invoke(query)  # qa_chain({"query": query}) 대신 invoke(query) 사용

    answer, source_documents = result['result'], result['source_documents']  # 결과에서 답변과 출처 문서 추출

    print("\n답변:")
    print(answer)  # 답변 출력

    print("\n출처 문서:")  # 출처 문서 출력
    for document in source_documents:
        # 각 문서의 파일 이름, 페이지 번호, 내용 일부 출력
        print(f"- 문서 이름: {document.metadata['source']}, 내용: {document.page_content[:200]}...") 