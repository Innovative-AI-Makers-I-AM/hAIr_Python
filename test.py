import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI  # OpenAI 챗 모델 import 변경
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 2단계: 임베딩 및 벡터 저장소 생성
embeddings = HuggingFaceInstructEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
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


# from sentence_transformers import SentenceTransformer, util
# from chromadb import PersistentClient
# from langchain.text_splitter import CharacterTextSplitter
# import os

# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAI
# from langchain.chains import retrieval, combine_documents
# from langchain.prompts import PromptTemplate
# from langchain_core.documents import Document  # 문서 객체 가져오기

# # 모델 및 ChromaDB 설정 함수
# def initialize_model():
#     # 모델 생성
#     return SentenceTransformer('upskyy/kf-deberta-multitask')

# def initialize_chromadb(collection_name='def_embeddings'):
#     client = PersistentClient()
#     collections = client.list_collections()
#     collection_names = [coll.name for coll in collections]
#     print(f'collection list: {collection_names}')

#     if collection_name in collection_names:
#         collection = client.get_collection(name=collection_name)
#         print(f'기존 컬렉션 {collection_name}을 사용합니다.')
#     else:
#         collection = client.create_collection(name=collection_name)
#         print(f'새 컬렉션 {collection_name}을 생성했습니다.')
    
#     return collection

# # TXT 파일을 읽어서 텍스트 데이터를 추출
# def read_txt(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()

# # 텍스트 임베딩 및 데이터 추가 함수
# def embed_and_add_to_chromadb(model, collection, file_paths):
#     all_texts = []

#     for file_path in file_paths:
#         text = read_txt(file_path)
#         all_texts.append(text)

#     embeddings = model.encode(all_texts)
#     ids = [str(i) for i in range(len(embeddings))]
#     collection.add(
#         documents=all_texts,
#         embeddings=embeddings,
#         ids=ids
#     )
#     print('\n임베딩 데이터 추가')
#     return embeddings, all_texts

# # 데이터 확인 함수
# def print_stored_data(collection):
#     stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
#     id_count = collection.count()
#     print(id_count)
#     print("\n컬렉션에 저장된 모든 데이터:")
#     for i in range(len(stored_data['ids'])):
#         print(f"ID: {stored_data['ids'][i]}")
#         print(f"Document: {stored_data['documents'][i]}")
#         print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
#         print(f"Metadata: {stored_data['metadatas'][i]}")
#         print("---")

# # 유사도 계산 함수
# def find_most_similar(query, model, collection, top_k=5):
#     query_embedding = model.encode(query)
#     stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
    
#     similarities = []

#     for i in range(len(stored_data['embeddings'])):
#         stored_embedding = stored_data['embeddings'][i]
#         similarity = util.pytorch_cos_sim(query_embedding, stored_embedding).item()
#         similarities.append((similarity, {
#             "id": stored_data['ids'][i],
#             "document": stored_data['documents'][i],
#             "metadata": stored_data['metadatas'][i],
#             "similarity": similarity
#         }))
    
#     # 유사도 순으로 정렬하고 상위 top_k 개 추출
#     similarities.sort(key=lambda x: x[0], reverse=True)
#     top_k_similarities = [item[1] for item in similarities[:top_k]]
    
#     return top_k_similarities

# # LLM 설정
# def initialize_llm(api_key):
#     return OpenAI(api_key=api_key, max_tokens=150, temperature=0.7, top_p=0.9)

# def generate_response(llm, context, question):
#     qa_prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="""
#         Context:
#           {context}
          
#         \n\nQuestion:
#           {question}

#         \n\nAnswer:
#         """
#         )
#     combine_docs_chain = combine_documents.create_stuff_documents_chain(llm, qa_prompt)
#     # qa_chain = retrieval.create_retrieval_chain(retriever=None, combine_docs_chain=combine_docs_chain)  # retriever=None for direct input
#     # Document 객체 목록 생성
#     documents = [Document(page_content=doc) for doc in context.split("\n")]
#     result = combine_docs_chain.invoke({"context": documents, "question": question})
#     return result


# if __name__ == "__main__":
#     # 모델 및 ChromaDB 초기화
#     model = initialize_model()
#     collection = initialize_chromadb()

#     # txt 파일 로드
#     txt_files = ['data/히피펌.txt', 'data/리프컷.txt']

#     # 컬렉션에 데이터가 없으면 TXT 파일에서 데이터를 로드하고 저장
#     if collection.count() == 0:
#         # 텍스트 임베딩 및 ChromaDB에 데이터 추가
#         embeddings, texts = embed_and_add_to_chromadb(model, collection, txt_files)
#     else:
#         print("컬렉션에 데이터가 이미 존재합니다.")

#     # 컬렉션에 저장된 데이터 확인
#     print_stored_data(collection)

#     # 예시 쿼리로 유사도 계산
#     query = "여름에 어울리는 히피펌 스타일 추천"
#     similar_texts = find_most_similar(query, model, collection)
#     print("\n유사한 텍스트:")
#     for text in similar_texts:
#         print(f"ID: {text['id']}")
#         print(f"Document: {text['document']}")
#         print(f"Similarity: {text['similarity']}")
#         print("---")
        
#         # LLM 초기화
#     llm = initialize_llm('open api key')

#     # 사용자 입력 받기
#     user_query = input("질문을 입력하세요: ")

#     # 유사한 텍스트 찾기
#     similar_texts = find_most_similar(user_query, model, collection)

#     # 컨텍스트 생성
#     context = "\n".join([text['document'] for text in similar_texts])

#     # LLM을 사용하여 응답 생성
#     response = generate_response(llm, context, user_query)

#     print("\nLLM 응답:")
#     print(response)




# from dotenv import load_dotenv
# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import Chroma
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
# from langchain.chains import retrieval, combine_documents
# from langchain_openai import OpenAI
# from langchain_huggingface import HuggingFaceEmbeddings

# import os

# # 환경 변수 로드
# load_dotenv()

# # 허깅페이스 임베딩 모델 초기화
# embeddings = HuggingFaceEmbeddings(model_name="upskyy/kf-deberta-multitask")

# # Chroma 벡터 저장소 초기화
# vector_store = Chroma(persist_directory="chroma", embedding_function=embeddings)

# # 문서 로더 초기화
# loader = TextLoader("./data/리프컷.txt")
# documents = loader.load()

# # 텍스트 분할
# text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
# texts = text_splitter.split_documents(documents)

# # 벡터 저장소에 문서 추가
# vector_store.add_documents(texts)

# # 리트리버 초기화
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# # OpenAI LLM 초기화
# llm = OpenAI(api_key="open api key", max_tokens=150, temperature=0.7, top_p=0.9)

# # RetrievalQA 체인 생성
# qa_chain = retrieval.create_retrieval_chain(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True
# )

# # 질문 실행
# question = "나 여름이어서 좀 상큼하게 머리 스타일 변경하고 싶어 그런데 나는 긴머리는 유지하고 싶어"
# result = qa_chain({"query": question})

# print("질문:", question)
# print("답변:", result["result"])
# print("\n관련 문서:")
# for i, doc in enumerate(result["source_documents"]):
#     print(f"문서 {i+1}:")
#     print(doc.page_content)
#     print("---")


# from sentence_transformers import SentenceTransformer, util
# from chromadb import PersistentClient
# from langchain.text_splitter import CharacterTextSplitter
# import os

# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAI
# from langchain.chains import retrieval, combine_documents
# from langchain.prompts import PromptTemplate
# from langchain_core.documents import Document  # 문서 객체 가져오기

# # TXT 파일을 읽어서 텍스트 데이터를 추출
# def read_txt(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()
    

# # 모델 및 ChromaDB 설정 함수
# def initialize_model():
#   # 모델 생성
#   return SentenceTransformer('upskyy/kf-deberta-multitask')

# # # 컬렉션 삭제 및 생성 함수
# # def reset_chromadb(collection_name='hairstyle_json_embeddings'):
# #     client = PersistentClient()
    
# #     try:
# #         client.delete_collection(name=collection_name)
# #         print(f'기존 컬렉션 {collection_name}을 삭제했습니다.')
# #     except Exception as e: 
# #         print(f'컬렉션 삭제 중 오류 발생: {e}')

# #     collection = client.create_collection(name=collection_name)
# #     print(f'새 컬렉션 {collection_name}을 생성했습니다.')
# #     return collection

# def initialize_chromadb(collection_name='abc_embeddings'):
#     client = PersistentClient()
#     collections = client.list_collections()
#     collection_names = [coll.name for coll in collections]
#     print(f'collection list :  {[collection_names]}')

#     if collection_name in collection_names:
#         collection = client.get_collection(name=collection_name)
#         print(f'기존 컬렉션 {collection_name}을 사용합니다.')
#     else:
#         collection = client.create_collection(name=collection_name)
#         print(f'새 컬렉션 {collection_name}을 생성했습니다.')
    
#     return collection

# # None 값을 빈 문자열로 변환하는 함수
# def clean_metadata(metadata):
#     for key, value in metadata.items():
#         if value is None:
#             metadata[key] = ""
#     return metadata

# # # chunk 로 데이터 자르기
# # def chunk_texts(texts, chunk_size=300, chunk_overlap=20):
# #     text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# #     chunked_texts = []
# #     for text in texts:
# #         chunks = text_splitter.split_text(text)
# #         chunked_texts.extend(chunks)
# #     return chunked_texts

# # 텍스트 임베딩 및 데이터 추가 함수
# def embed_and_add_to_chromadb(model, collection, file_paths):
#     all_chunks = []
#     # all_metadatas = []

#     for file_path in file_paths:
#         text = read_txt(file_path)
#         # chunks = chunk_texts([text])
#         all_chunks.extend(text)
#         # 각 chunk에 파일명을 메타데이터로 포함
#         # style = os.path.basename(file_path).split('.')[0]  # 파일명에서 확장자 제거
#         # metadatas = [{'description': chunk, 'style': style} for chunk in chunks]
#         # all_metadatas.extend(metadatas)

#     embeddings = model.encode(all_chunks)
#     ids = [str(i) for i in range(len(embeddings))]
#     collection.add(
#         documents=all_chunks,
#         embeddings=embeddings,
#         # metadatas=all_metadatas,
#         ids=ids
#     )
#     print('\n임베딩 데이터 추가')
#     return embeddings, all_chunks


# # 데이터 확인 함수
# def print_stored_data(collection):
#     stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
#     id_count = collection.count()
#     print(id_count)
#     print("\n hairstyle_embeddings 컬렉션에 저장된 모든 데이터:")
#     for i in range(len(stored_data['ids'])):
#         print(f"ID: {stored_data['ids'][i]}")
#         print(f"Document: {stored_data['documents'][i]}")
#         print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
#         print(f"Metadata: {stored_data['metadatas'][i]}")
#         print("---")

# # 유사도 계산 함수
# def find_most_similar(query, model, collection, top_k=5):
#     query_embedding = model.encode(query)
#     stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
    
#     similarities = []

#     for i in range(len(stored_data['embeddings'])):
#         stored_embedding = stored_data['embeddings'][i]
#         similarity = util.pytorch_cos_sim(query_embedding, stored_embedding).item()
#         similarities.append((similarity, {
#             "id": stored_data['ids'][i],
#             "document": stored_data['documents'][i],
#             "metadata": stored_data['metadatas'][i],
#             "similarity": similarity
#         }))
    
#     # 유사도 순으로 정렬하고 상위 top_k 개 추출
#     similarities.sort(key=lambda x: x[0], reverse=True)
#     top_k_similarities = [item[1] for item in similarities[:top_k]]
    
#     return top_k_similarities

# # # 리트리버 초기화
# # def initialize_retriever(collection_name, model):
# #     class EmbeddingFunction:
# #         def __init__(self, model):
# #             self.model = model

# #         def embed_query(self, query):
# #             return self.model.encode([query])[0].tolist()
        
# #     embedding_function = EmbeddingFunction(model)
# #     vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding_function)
# #     retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
# #     print(f"리트리버 초기화 완료: {retriever}")
# #     return retriever, embedding_function

# # # 검색 함수
# # def find_most_similar(query, retriever):
# #     print(f"검색 쿼리: {query}")
# #     # query_embedding = embedding_function.embed_query(query)
# #     # print(f"쿼리 임베딩: {query_embedding}")
# #     retrieved_docs = retriever.invoke({"query":query})

# #     try:
# #         retrieved_docs = retriever.invoke({"query": query})

# #         print(f"검색 결과: {retrieved_docs}")

# #         if retrieved_docs:
# #             print(f"검색 결과: {retrieved_docs}")
# #             return retrieved_docs[0]
# #         else:
# #             return None
        
# #     except Exception as e:
# #         print(f"Retrieval Error: {e}")
# #         return None


# # LLM 설정
# def initialize_llm(api_key):
#     return OpenAI(api_key=api_key, max_tokens=150, temperature=0.7, top_p=0.9)

# def generate_response(llm, context, question):
#     qa_prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="""
#         Context:
#           {context}
          
#         \n\nQuestion:
#           {question}

#         \n\nAnswer:
#         """
#         )
#     combine_docs_chain = combine_documents.create_stuff_documents_chain(llm, qa_prompt)
#     # qa_chain = retrieval.create_retrieval_chain(retriever=None, combine_docs_chain=combine_docs_chain)  # retriever=None for direct input
#     # Document 객체 목록 생성
#     documents = [Document(page_content=doc) for doc in context.split("\n")]
#     result = combine_docs_chain.invoke({"context": documents, "question": question})
#     return result




# if __name__ == "__main__":
#     # 모델 및 ChromaDB 초기화
#     model = initialize_model()
#     # collection = reset_chromadb()
#     collection = initialize_chromadb()

#     # txt 파일 로드
#     txt_files = ['data/리프컷.txt', 'data/히피펌.txt']
#     # texts = [read_txt(file) for file in txt_files]

#     # 컬렉션에 데이터가 없으면 JSON 파일에서 데이터를 로드하고 저장
#     if collection.count() == 0:

#         # 텍스트 임베딩 및 ChromaDB에 데이터 추가
#         embeddings, texts = embed_and_add_to_chromadb(model, collection, txt_files)
#     else:
#         print("컬렉션에 데이터가 이미 존재합니다.")

#     # 컬렉션에 저장된 데이터 확인
#     print_stored_data(collection)

#     # # 예제 쿼리 검색
#     # query = "나 여름이어서 좀 상큼하게 머리 스타일 변경하고 싶어 그런데 나는 긴머리는 유지하고 싶어"
#     # most_similar_hairstyles = find_most_similar(query, model, collection)
#     # if most_similar_hairstyles:
#     #     print(f"가장 유사한 스타일:")
#     #     for idx, hairstyle in enumerate(most_similar_hairstyles):
#     #         print(f"\n유사도 {idx + 1}: {hairstyle}")

#     #     context = "\n".join([f"{hairstyle['metadata']['description']}: {hairstyle['document']}" for hairstyle in most_similar_hairstyles])
#     #     question = """
#     #     쓸데없는 꾸밈말은 삼가주세요.
#     #     사용자 요청 사항에 맞게 머리 스타일에 대해서 추천해주세요.
#     #       그리고 요청 사항과 맞는 또다른 머리 스타일이 존재한다면 그것도 알려주세요.
#     #       그리고 답변은 100자 이내로 요약해서 답해주세요."""
        
#     #     llm = initialize_llm(api_key='open api key')  # 여기에 OpenAI API 키를 입력하세요.
#     #     response = generate_response(llm, context, question)
#     #     print(f"\nAnswer: {response}")
#     # else:
#     #     print(f"{query}와 유사한 스타일을 찾을 수 없습니다.")


#     # # 리트리버 초기화
#     # retriever, embedding_function = initialize_retriever(collection.name, model)

#     # # 예제 쿼리 검색
#     # query = "레이어드컷"
#     # most_similar_hairstyle = find_most_similar(query, retriever)
#     # if most_similar_hairstyle:
#     #     print(f"가장 유사한 스타일: {most_similar_hairstyle}")
#     # else:
#     #     print(f"{query}와 유사한 스타일을 찾을 수 없습니다.")

#     # # RAG 설정 및 답변 생성
#     # llm = OpenAI(api_key='openai api key', model='gpt-4o')
#     # # llm = OpenAI(api_key='open-api-key')
#     # qa_prompt = PromptTemplate(input_variables=["context", "question"], template="Context: {context}\n\nQuestion: {question}\n\nAnswer:")
#     # combine_docs_chain = combine_documents.create_stuff_documents_chain(llm, qa_prompt)
#     # qa_chain = retrieval.create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

#     # # 질문에 대한 답변 생성
#     # question = "레이어드컷에 대해 알려줘"
#     # if most_similar_hairstyle:
#     #   result = qa_chain.invoke({"context": most_similar_hairstyle.page_content, "question": question})
#     #   print(f"\nAnswer: {result}")
#     # else:
#     #     print(f"{question}의 질문과 유사한 스타일을 찾을 수 없어 답변을 생성할 수 없습니다.")



    