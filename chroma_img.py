# from PIL import Image
import numpy as np
import torch
# from transformers import ViTImageProcessor, ViTModel
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, util

def initialize_collection():
    # ChromaDB 설정 및 클라이언트 생성
    client = PersistentClient()

    # 컬렉션 생성 또는 기존 컬렉션 가져오기
    collection_name = "text_embeddings"
    collections = client.list_collections()
    collection_names = [coll.name for coll in collections]

    if collection_name in collection_names:
        collection = client.get_collection(name=collection_name)
        print(f"기존 컬렉션 {collection_name}을 사용합니다.")
    else:
        collection = client.create_collection(name=collection_name)
        print(f"새 컬렉션 {collection_name}을 생성했습니다.")

    # 모델 생성
    model = SentenceTransformer('upskyy/kf-deberta-multitask')

    # 헤어 스타일 데이터
    hairstyle_data = [
        {
            "style_name": "레이어드컷",
            "gender": ["남", "여"],
            "hair_length": ["롱", "미디움"],
            "bangs": ["유", "무"],
            "face_shape": ["둥근형", "계란형", "긴형", "네모형", "다이아몬드형"],
            "description": "레이어드 컷은 여러 층으로 나누어져 있어 자연스러운 볼륨감과 움직임을 줍니다. 긴 머리와 중간 길이의 머리에 잘 어울리며, 다양한 얼굴형에 적합합니다. 포멀한 분위기를 줄 수 있다",
            "image_link": "https://contents-cdn.viewus.co.kr/image/2024/03/CP-2022-0277/image-f3905fb3-8f54-4e83-be13-444dfdb3acd3.jpeg"
        },
        {
            "style_name": "드롭컷",
            "gender": ["남"],
            "hair_length": ["숏"],
            "bangs": ["유, 무"],
            "face_shape": ["계란형", "둥근형"],
            "description": "아이비리그 컷에서 파생된 헤어컷으로 사이드 부분만 앞머리를 내리고 센터 라인은 올리는 스타일 짧은 머리 스타일로 시원한 인상을 주고, 머리카락 텍스처감을 살리는 스타일링으로 트렌디한 여름 헤어를 찾으시는 분들께 안성맞춤이다",
            "image_link": "https://example.com/images/bob_cut.jpg"
        },
        {
            "style_name": "리프컷",
            "gender": ["남", "여"],
            "hair_length": ["롱, 미디움"],
            "bangs": ["유", "무"],
            "face_shape": ["긴형", "둥근형", "네모형", "다이아몬드형"],
            "description": "전체적인 디자인이 나뭇잎을 닮은 스타일 이름처럼 앞머리부터 옆머리까지 둥글게 넘어가도록 스타일링하는 게 포인트. 남자 리프컷은 어떤 기장에서도 포멀한 분위기를 연출. 베이직한 리프컷은 장발 스타일에 속함. 장발 기장은 보통 덥수룩한 느낌이 들지만 끝부분을 질감 처리한 리프컷으로 가볍게 연출하면 긴 머리에서도 깔끔해 보인다. 짧은 기장에서는 세미 리프컷으로 좀 더 슬림 하고 세련된 분위기로 스타일링이 가능. 여자 리프컷은 목 선이 드러나는 숏컷 스타일로, 각자 가진 이미지에 따라 다양한 매력을 보여 줄 수 있다. 보이시한 느낌이 강한 분들은 층의 단차를 크게 내주세요. 시크하고 도시적인 분위기를 준다. 러블리하고 청순한 이미지라면 전체적인 디자인을 둥글게 정리해 주세요. 뒷부분 볼륨도 넣어 주면 맑고 깨끗한 인상이 돋보인다.",
            "image_link": "https://mud-kage.kakao.com/dn/7NfgO/btszgZ0enEM/AJTXbaz6ymG7tQe8B6Crpk/img_750.jpg"
        }
    ]

    # JSON 데이터를 텍스트로 변환
    texts = []
    metadatas = []

    for i, data in enumerate(hairstyle_data):
        text_data = f"""
        설명: {data['description']}
        """
        texts.append(text_data)
        metadatas.append({
            "style_name": data['style_name'],
            "gender": ', '.join(data['gender']),
            "hair_length": ', '.join(data['hair_length']),
            "bangs": ', '.join(data['bangs']),
            "face_shape": ', '.join(data['face_shape']),
            "description": data['description'],
            "image_link": data['image_link']
        })

    embeddings = model.encode(texts)  # input 데이터 임베딩으로 변환
    embeddings = [embedding.tolist() for embedding in embeddings]  # 임베딩값 list 형식으로 받음

    # 임베딩 데이터 추가
    ids = [str(i) for i in range(len(embeddings))]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # 컬렉션에 저장된 값 확인
    print("\n컬렉션에 저장된 모든 데이터:")
    stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
    for i in range(len(stored_data['ids'])):
        print(f"ID: {stored_data['ids'][i]}")
        print(f"Document: {stored_data['documents'][i]}")
        print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
        print(f"Metadata: {stored_data['metadatas'][i]}")
        print("---")
    
    return collection, model

collection, model = initialize_collection()

def update_collection(collection, model, style_name_to_update, new_description=None, new_image_link=None):
    # 컬렉션에 저장된 값 확인
    stored_data = collection.get(include=["documents", "embeddings", "metadatas"])

    # 기존 데이터를 검색하여 ID를 찾기
    update_id = None
    for i in range(len(stored_data['ids'])):
        if stored_data['metadatas'][i]['style_name'] == style_name_to_update:
            update_id = stored_data['ids'][i]
            current_metadata = stored_data['metadatas'][i]
            break

    if update_id is not None:
        # 기존 메타데이터에서 원하는 부분만 수정
        if new_description:
            current_metadata["description"] = new_description
        if new_image_link:
            current_metadata["image_link"] = new_image_link

        # 업데이트할 데이터를 준비
        update_texts = [current_metadata["description"]]
        update_embeddings = model.encode(update_texts)
        update_embeddings = [embedding.tolist() for embedding in update_embeddings]

        # 기존 데이터 업데이트
        collection.update(
            ids=[update_id],
            documents=update_texts,
            embeddings=update_embeddings,
            metadatas=[current_metadata]
        )

        print('업데이트가 완료되었습니다.')
    else:
        print(f"Style name '{style_name_to_update}'에 해당하는 데이터를 찾을 수 없습니다.")

    print("\n업데이트 후 컬렉션에 저장된 모든 데이터:")
    stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
    for i in range(len(stored_data['ids'])):
        print(f"ID: {stored_data['ids'][i]}")
        print(f"Document: {stored_data['documents'][i]}")
        print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
        print(f"Metadata: {stored_data['metadatas'][i]}")
        print("---")

# 예시 업데이트 호출
update_collection(collection, model, "레이어드컷", new_description="수정 테스트 분리", new_image_link="https://newlink.com/newimage.jpeg")

class Retriever:
    def __init__(self, model, collection_name='text_embeddings'):
        self.model = model
        self.client = PersistentClient()
        self.collection_name = collection_name
        collections = self.client.list_collections()
        collection_names = [coll.name for coll in collections]

        if collection_name in collection_names:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"기존 컬렉션 {collection_name}을 사용합니다.")
        else:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"새 컬렉션 {collection_name}을 생성했습니다.")

    def retrieve(self, query, top_k=2):
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array(query_embedding).astype(np.float64)

        # 저장된 임베딩과의 유사도 계산
        stored_data = self.collection.get(include=["documents", "embeddings", "metadatas"])
        scores = []
        for idx, stored_embedding in enumerate(stored_data['embeddings']):
            stored_embedding = np.array(stored_embedding).astype(np.float64)
            similarity = util.cos_sim(torch.tensor(query_embedding), torch.tensor(stored_embedding))
            scores.append((similarity.item(), idx))

        scores.sort(reverse=True, key=lambda x: x[0])
        top_results = scores[:top_k]

        results = []
        for score, idx in top_results:
            results.append({
                "similarity": score,
                "id": stored_data['ids'][idx],
                "document": stored_data['documents'][idx],
                "metadata": stored_data['metadatas'][idx]
            })

        return results

# Retriever 인스턴스 생성
retriever = Retriever(model)

# 예제 쿼리
query = input('무슨 스타일 원하시나요? ')
results = retriever.retrieve(query)

for result in results:
    print(f"\n유사도: {result['similarity']}")
    print(f"ID: {result['id']}")
    print(f"Document: {result['document']}")
    print(f"Metadata: {result['metadata']}")
    print("---")



##############################

# from chromadb import PersistentClient
# from sentence_transformers import SentenceTransformer

# import os
# from langchain.vectorstores import Chroma
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

# # OpenAI API 키 설정
# # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# # ChromaDB 설정 및 클라이언트 생성
# client = PersistentClient()

# # 컬렉션 생성 또는 기존 컬렉션 가져오기
# collection_name = "text_embeddings"
# collections = client.list_collections()
# collection_names = [coll.name for coll in collections]

# if collection_name in collection_names:
#     collection = client.get_collection(name=collection_name)
#     print(f"기존 컬렉션 {collection_name}을 사용합니다.")
# else:
#     collection = client.create_collection(name=collection_name)
#     print(f"새 컬렉션 {collection_name}을 생성했습니다.")

# model = SentenceTransformer('upskyy/kf-deberta-multitask')

# # input 데이터
# hairstyle_data = [
#     {
#       "style_name": "레이어드 컷",
#       "gender": ["남", "여"],
#       "hair_length": ["롱", "미디움"],
#       "bangs": ["유", "무"],
#       "face_shape": ["둥근형", "계란형", "긴형", "네모형", "다이아몬드형"],
#       "description": "레이어드 컷은 여러 층으로 나누어져 있어 자연스러운 볼륨감과 움직임을 줍니다. 긴 머리와 중간 길이의 머리에 잘 어울리며, 다양한 얼굴형에 적합합니다.",
#       "image_link": "https://contents-cdn.viewus.co.kr/image/2024/03/CP-2022-0277/image-f3905fb3-8f54-4e83-be13-444dfdb3acd3.jpeg"
#     },
#     {
#       "style_name": "단발 컷",
#       "gender": ["여"],
#       "hair_length": ["단발"],
#       "bangs": ["유"],
#       "face_shape": ["계란형", "둥근형"],
#       "description": "단발 컷은 깔끔하고 세련된 스타일로, 계란형과 둥근형 얼굴형에 잘 어울립니다.",
#       "image_link": "https://example.com/images/bob_cut.jpg"
#     }
# ]

# # JSON 데이터를 텍스트로 변환
# texts = []
# metadatas = []

# for i, data in enumerate(hairstyle_data):
#     text_data = f"""
#     설명: {data['description']}
#     """
#     texts.append(text_data)
#     metadatas.append({
#         "style_name": data['style_name'],
#         "gender":', '.join(data['gender']),
#         "hair_length":', '.join(data['hair_length']),
#         "bangs":', '.join(data['bangs']),
#         "face_shape":  ', '.join(data['face_shape']),
#         "description": data['description'],
#         "image_link": data['image_link']
#     })

# embeddings = model.encode(texts) # input 데이터 임베딩으로 변환
# embeddings = [embedding.tolist() for embedding in embeddings] # 임베딩값 list 형식으로 받음


# # 임베딩 데이터 추가
# ids = [str(i+1) for i in range(len(embeddings))]
# collection.add(
#     documents=texts,
#     embeddings=embeddings,
#     metadatas=metadatas,
#     ids=ids
# )

# # 컬렉션에 저장된 값 확인
# print("\n컬렉션에 저장된 모든 데이터:")
# stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
# for i in range(len(stored_data['ids'])):
#     print(f"ID: {stored_data['ids'][i]}")
#     print(f"Document: {stored_data['documents'][i]}")
#     print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
#     print(f"Metadata: {stored_data['metadatas'][i]}")
#     print("---")


# # Embedding Function Wrapper
# class EmbeddingFunction:
#     def __init__(self, model):
#         self.model = model

#     def embed_query(self, query):
#         return self.model.encode([query])[0].tolist()
    
# embedding_function = EmbeddingFunction(model)

# # 검색기 설정
# vectorstore = Chroma(collection_name='text_embeddings', embedding_function=embedding_function)
# retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})

# # 질문하기
# query = input('Ask your question: ')

# # 검색 실행
# retrieved_docs = retriever.get_relevant_documents(query)

# # 결과 출력
# print("Recommended hairstyles:")
# if retrieved_docs:
#     print('Recommended hairstyle : ')
#     doc = retrieved_docs[0]
#     print(f"Style name: {doc.metadata['style_name']}")
#     print(f"Description: {doc.metadata['description']}")
#     print(f"Image link: {doc.metadata['image_link']}")
# else:
#     print("No matching hairstyle found.")

# # RetrievalQA 체인 설정
# llm = OpenAI(temperature=0)
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# # 질문에 대한 답변 생성
# result = qa({"query": query})
# print(f'\nAnswer : {result["result"]}')




###################################

# ### 이미지를 벡터화하여 Chroma DB 에 임베딩값 저장 ###
# from chromadb import PersistentClient
# from sentence_transformers import SentenceTransformer

# import os
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAI
# # from langchain.runnables import RunnableSequence
# from langchain.prompts import PromptTemplate
# # from langchain.chains import retrieval_qa
# from langchain.chains import retrieval_qa, retrieval

# # OpenAI API 키 설정
# # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# # ChromaDB 설정 및 클라이언트 생성
# client = PersistentClient()

# # 컬렉션 생성 또는 기존 컬렉션 가져오기
# collection_name = "text_embeddings"
# collections = client.list_collections()
# collection_names = [coll.name for coll in collections]

# if collection_name in collection_names:
#     collection = client.get_collection(name=collection_name)
#     print(f"기존 컬렉션 {collection_name}을 사용합니다.")
# else:
#     collection = client.create_collection(name=collection_name)
#     print(f"새 컬렉션 {collection_name}을 생성했습니다.")

# model = SentenceTransformer('upskyy/kf-deberta-multitask')

# # input 데이터
# hairstyle_data = [
#     {
#       "style_name": "레이어드 컷",
#       "gender": ["남", "여"],
#       "hair_length": ["롱", "미디움"],
#       "bangs": ["유", "무"],
#       "face_shape": ["둥근형", "계란형", "긴형", "네모형", "다이아몬드형"],
#       "description": "레이어드 컷은 여러 층으로 나누어져 있어 자연스러운 볼륨감과 움직임을 줍니다. 긴 머리와 중간 길이의 머리에 잘 어울리며, 다양한 얼굴형에 적합합니다.",
#       "image_link": "https://contents-cdn.viewus.co.kr/image/2024/03/CP-2022-0277/image-f3905fb3-8f54-4e83-be13-444dfdb3acd3.jpeg"
#     },
#     {
#       "style_name": "단발 컷",
#       "gender": ["여"],
#       "hair_length": ["단발"],
#       "bangs": ["유"],
#       "face_shape": ["계란형", "둥근형"],
#       "description": "단발 컷은 깔끔하고 세련된 스타일로, 계란형과 둥근형 얼굴형에 잘 어울립니다.",
#       "image_link": "https://example.com/images/bob_cut.jpg"
#     }
# ]

# # JSON 데이터를 텍스트로 변환
# texts = []
# metadatas = []

# for i, data in enumerate(hairstyle_data):
#     text_data = f"""
#     설명: {data['description']}
#     """
#     texts.append(text_data)
#     metadatas.append({
#         "style_name": data['style_name'],
#         "gender":', '.join(data['gender']),
#         "hair_length":', '.join(data['hair_length']),
#         "bangs":', '.join(data['bangs']),
#         "face_shape":  ', '.join(data['face_shape']),
#         "description": data['description'],
#         "image_link": data['image_link']
#     })

# embeddings = model.encode(texts) # input 데이터 임베딩으로 변환
# embeddings = [embedding.tolist() for embedding in embeddings] # 임베딩값 list 형식으로 받음


# # 임베딩 데이터 추가
# ids = [str(i+1) for i in range(len(embeddings))]
# collection.add(
#     documents=texts,
#     embeddings=embeddings,
#     metadatas=metadatas,
#     ids=ids
# )

# # 컬렉션에 저장된 값 확인
# print("\n컬렉션에 저장된 모든 데이터:")
# stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
# for i in range(len(stored_data['ids'])):
#     print(f"ID: {stored_data['ids'][i]}")
#     print(f"Document: {stored_data['documents'][i]}")
#     print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
#     print(f"Metadata: {stored_data['metadatas'][i]}")
#     print("---")

# # Embedding Function Wrapper
# class EmbeddingFunction:
#     def __init__(self, model):
#         self.model = model

#     def embed_query(self, query):
#         return self.model.encode([query])[0].tolist()
    
# embedding_function = EmbeddingFunction(model)

# # 검색기 설정
# vectorstore = Chroma(collection_name='text_embeddings', embedding_function=embedding_function)
# retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})

# # 질문하기
# query = input('Ask your question: ')

# # # 텍스트 임베딩 생성
# # question_embedding = model.encode([question])[0]

# # 검색 실행
# retrieved_docs = retriever.get_relevant_documents(query=query)

# # 결과 출력
# print("Recommended hairstyles:")
# if retrieved_docs:
#     print('Recommended hairstyle : ')
#     doc = retrieved_docs[0]
#     print(f"Style name: {doc.metadata['style_name']}")
#     print(f"Description: {doc.metadata['description']}")
#     print(f"Image link: {doc.metadata['image_link']}")
# else:
#     print("No matching hairstyle found.")

# # for doc in retrieved_docs:
# #     print(f"Recommended style name: {doc.metadata['style_name']}")
# #     print(f"Description: {doc.metadata['description']}")
# #     print(f"Image link: {doc.metadata['image_link']}")
# #     print('---')

# # RetrievalQA 체인 설정
# # LLM 설정
# llm = OpenAI(temperature=0)
# # qa_prompt = PromptTemplate(input_variables=["context", "question"], template="Context: {context}\n\nQuestion: {question}\n\nAnswer:")
# # llm_chain = RunnableSequence([qa_prompt, llm])
# qa = retrieval.create_retrieval_chain(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# # 질문에 대한 답변 생성
# result = qa({"query": query})
# print(f'\nAnswer : {result["result"]}')


###############################################################


# from PIL import Image
# import numpy as np
# import torch
# from transformers import ViTImageProcessor, ViTModel
# from chromadb import PersistentClient
# from sentence_transformers import SentenceTransformer, util

# # ChromaDB 설정 및 클라이언트 생성
# client = PersistentClient()

# # 컬렉션 생성 또는 기존 컬렉션 가져오기
# collection_name = "text_embeddings"
# collections = client.list_collections()
# collection_names = [coll.name for coll in collections]

# if collection_name in collection_names:
#     collection = client.get_collection(name=collection_name)
#     print(f"기존 컬렉션 {collection_name}을 사용합니다.")
# else:
#     collection = client.create_collection(name=collection_name)
#     print(f"새 컬렉션 {collection_name}을 생성했습니다.")

# model = SentenceTransformer('upskyy/kf-deberta-multitask')

# # input 데이터
# hairstyle_data = [
#     {
#         "style_name": "레이어드 컷",
#         "gender": ["남", "여"],
#         "hair_length": ["롱", "미디움"],
#         "bangs": ["유", "무"],
#         "face_shape": ["둥근형", "계란형", "긴형", "네모형", "다이아몬드형"],
#         "description": "레이어드 컷은 여러 층으로 나누어져 있어 자연스러운 볼륨감과 움직임을 줍니다. 긴 머리와 중간 길이의 머리에 잘 어울리며, 다양한 얼굴형에 적합합니다.",
#         "image_link": "https://contents-cdn.viewus.co.kr/image/2024/03/CP-2022-0277/image-f3905fb3-8f54-4e83-be13-444dfdb3acd3.jpeg"
#     },
#     {
#         "style_name": "단발 컷",
#         "gender": ["여"],
#         "hair_length": ["단발"],
#         "bangs": ["유"],
#         "face_shape": ["계란형", "둥근형"],
#         "description": "단발 컷은 깔끔하고 세련된 스타일로, 계란형과 둥근형 얼굴형에 잘 어울립니다.",
#         "image_link": "https://example.com/images/bob_cut.jpg"
#     }
# ]

# # JSON 데이터를 텍스트로 변환
# texts = []
# metadatas = []

# for i, data in enumerate(hairstyle_data):
#     text_data = f"""
#     설명: {data['description']}
#     """
#     texts.append(text_data)
#     metadatas.append({
#         "style_name": data['style_name'],
#         "gender": ', '.join(data['gender']),
#         "hair_length": ', '.join(data['hair_length']),
#         "bangs": ', '.join(data['bangs']),
#         "face_shape": ', '.join(data['face_shape']),
#         "description": data['description'],
#         "image_link": data['image_link']
#     })

# embeddings = model.encode(texts)  # input 데이터 임베딩으로 변환
# embeddings = [embedding.tolist() for embedding in embeddings]  # 임베딩값 list 형식으로 받음

# # 임베딩 데이터 추가
# ids = [str(i) for i in range(len(embeddings))]
# collection.add(
#     documents=texts,
#     embeddings=embeddings,
#     metadatas=metadatas,
#     ids=ids
# )

# # 컬렉션에 저장된 값 확인
# print("\n컬렉션에 저장된 모든 데이터:")
# stored_data = collection.get(include=["documents", "embeddings", "metadatas"])
# for i in range(len(stored_data['ids'])):
#     print(f"ID: {stored_data['ids'][i]}")
#     print(f"Document: {stored_data['documents'][i]}")
#     print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")
#     print(f"Metadata: {stored_data['metadatas'][i]}")
#     print("---")

# # 입력 쿼리 임베딩 및 유사도 비교
# def find_similar_styles(query, top_k=1):
#     query_embedding = model.encode([query])[0]  # 쿼리를 임베딩으로 변환
#     query_embedding = np.array(query_embedding).astype(np.float64)  # 쿼리 임베딩을 double 타입으로 변환

#     # 저장된 임베딩과의 유사도 계산
#     scores = []
#     for idx, stored_embedding in enumerate(stored_data['embeddings']):
#         stored_embedding = np.array(stored_embedding).astype(np.float64)  # 저장된 임베딩을 double 타입으로 변환
#         similarity = util.cos_sim(torch.tensor(query_embedding), torch.tensor(stored_embedding))
#         scores.append((similarity.item(), idx))

#     # 유사도 순으로 정렬하여 top_k 결과 반환
#     scores.sort(reverse=True, key=lambda x: x[0])
#     top_results = scores[:top_k]

#     for score, idx in top_results:
#         print(f"\n유사도: {score}")
#         print(f"ID: {stored_data['ids'][idx]}")
#         print(f"Document: {stored_data['documents'][idx]}")
#         print(f"Metadata: {stored_data['metadatas'][idx]}")
#         print("---")

# # 예제 쿼리
# query = "계란형 얼굴에 잘 어울리는 컷을 추천해줘"
# find_similar_styles(query)






























# import requests
# from bs4 import BeautifulSoup
# # from langchain_community.vectorstores import Chroma
# # from langchain_community.embeddings import OpenAIEmbeddings
# import os

# from urllib.request import urlopen
# import io

# # persist_directory = 'db'
# # embedding = OpenAIEmbeddings()
# # vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# # 크롤링 할 base url
# base_url = 'https://hairshop.kakao.com/search/style'

# # 크롤링 할 카테고리 리스트
# categories = ['CUT', 'PERM', 'COLOR', 'STYLING','MAKEUP']

# tags = ['레이어드컷', '허쉬컷']

# # 이미지를 저장할 디렉토리 생성
# if not os.path.exists('images'):
#   os.makedirs('images')

# for category in categories:
#   for tag in tags:
#     url =f'{base_url}?category={category}&gender=FEMALE&tags={tag}'

#     # Http 요청 보내기
#     response = requests.get(url)
#     if response.status_code != 200:
#       print(f'Failed to retrieve data from {url}')
#       continue

#     # html 파싱
#     soup = BeautifulSoup(response.text, 'html.parser')

#     image_tags = soup.find_all('img', {'src':True, 'class':'img_g'})

#     for idx, image_tag in enumerate(image_tags, start=1):
#       image_url = image_tag['src']

#       if not image_url.startswith('http'):
#         if image_url.startswith('//'):
#           image_url = 'https:' + image_url
#         else:
#           image_url = 'https://' + image_url
      
#       image_response = requests.get(image_url)
#       if image_response.status_code == 200:
#         image_name = f'{category}_{tag}_{idx}.jpg'
#         image_path = os.path.join('images', category, tag)
#         os.makedirs(image_path, exist_ok=True)
#         image_file_path = os.path.join(image_path, image_name)

#         # 이미지 저장
#         with open(image_file_path, 'wb') as file:
#             file.write(image_response.content)
#             print(f'Downloaded image: {image_name}')

#       else:
#         print(f'Failed to download image from {image_url}')

# print('Image downloading completed.')






####################
# import sys
# import os

# # Ensure the parent directory is in the path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import requests
# from bs4 import BeautifulSoup
# import time
# from datetime import datetime
# import logging
# from backend.config import recipes_collection
# from backend.headers import get_random_headers
# from backend.proxies import get_random_proxy, remove_invalid_proxy

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# def get_recipe_urls(start_page=500, max_urls=5000):
#     """레시피 URL을 가져오는 함수"""
#     recipe_urls = []
#     page = start_page
#     while len(recipe_urls) < max_urls:
#         url = f'https://www.10000recipe.com/recipe/list.html?order=reco&page={page}'
#         headers = get_random_headers()
#         proxy = get_random_proxy()
#         try:
#             start_time = time.time()
#             if proxy:
#                 response = requests.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=30)
#             else:
#                 response = requests.get(url, headers=headers, timeout=30)
#             elapsed_time = time.time() - start_time
#             logging.info(f"Fetched {url} in {elapsed_time:.2f} seconds using proxy {proxy}")
#             response.raise_for_status()
#         except requests.exceptions.RequestException as e:
#             logging.error(f"Error fetching {url} using proxy {proxy}: {e}")
#             if proxy:
#                 remove_invalid_proxy(proxy)
#                 logging.info("Retrying without proxy...")
#                 try:
#                     response = requests.get(url, headers=headers, timeout=30)
#                     elapsed_time = time.time() - start_time
#                     logging.info(f"Fetched {url} in {elapsed_time:.2f} seconds without proxy")
#                     response.raise_for_status()
#                 except requests.exceptions.RequestException as e:
#                     logging.error(f"Error fetching {url} without proxy: {e}")
#                     break
#             else:
#                 break

#         soup = BeautifulSoup(response.text, 'html.parser')
#         items = soup.select('li.common_sp_list_li')
#         if not items:
#             logging.info(f"No more items found on page {page}. Ending search.")
#             break

#         for item in items:
#             recipe_url = 'https://www.10000recipe.com' + item.select('a.common_sp_link')[0].get('href')
#             if recipe_url not in recipe_urls:
#                 recipe_urls.append(recipe_url)
#             if len(recipe_urls) >= max_urls:
#                 break

#         page += 1
#         time.sleep(1)
#     logging.info(f"Total recipe URLs fetched: {len(recipe_urls)}")
#     return recipe_urls

# def extract_platform(url):
#     """URL에서 플랫폼(도메인 이름)을 추출하는 함수"""
#     platform = url.split('//')[1].split('/')[0].split('.')[1]
#     return platform

# def scrape_recipe(url, new_recipes_count, retries=5):
#     """레시피 데이터를 크롤링하여 MongoDB에 저장하는 함수"""
#     if recipes_collection.find_one({"oriUrl": url}):
#         logging.info(f"Skipping {url}, already in database")
#         return None

#     headers = get_random_headers()
#     proxy = get_random_proxy()

#     for attempt in range(retries):
#         try:
#             start_time = time.time()
#             if proxy:
#                 response = requests.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=30)
#             else:
#                 response = requests.get(url, headers=headers, timeout=30)
#             elapsed_time = time.time() - start_time
#             logging.info(f"Scraped {url} in {elapsed_time:.2f} seconds using proxy {proxy}")
#             response.raise_for_status()
#             break
#         except requests.exceptions.RequestException as e:
#             logging.warning(f"Error scraping {url} using proxy {proxy}: {e}, Retrying... ({retries - attempt - 1} retries left)")
#             time.sleep(5)
#             if proxy:
#                 remove_invalid_proxy(proxy)
#             proxy = get_random_proxy()
#             if attempt == retries - 1:
#                 logging.error(f"Failed to scrape {url} after {retries} attempts")
#                 return None

#     soup = BeautifulSoup(response.text, 'lxml')
#     summary_div = soup.find('div', class_='view2_summary st3')
#     title = summary_div.find('h3').get_text(strip=True) if summary_div and summary_div.find('h3') else None
#     if not title:
#         logging.info(f"Skipping {url}, title is empty")
#         return None

#     serving_info_div = soup.find('div', class_='view2_summary_info')
#     servings = serving_info_div.find('span', class_='view2_summary_info1').get_text(strip=True) if serving_info_div and serving_info_div.find('span', 'view2_summary_info1') else None
#     cookingTime = serving_info_div.find('span', class_='view2_summary_info2').get_text(strip=True) if serving_info_div and serving_info_div.find('span', 'view2_summary_info2') else None
#     level = serving_info_div.find('span', class_='view2_summary_info3').get_text(strip=True) if serving_info_div and serving_info_div.find('span', 'view2_summary_info3') else None

#     ingredients = []
#     for ingredient_section in soup.find_all('div', class_='ready_ingre3'):
#         for li in ingredient_section.find_all('li'):
#             name = li.find('div', class_='ingre_list_name').get_text(strip=True) if li.find('div', 'ingre_list_name') else None
#             amount = li.find('span', class_='ingre_list_ea').get_text(strip=True) if li.find('span', 'ingre_list_ea') else None
#             ingredients.append(f"{name} - {amount}")

#     instructions = []
#     for step_section in soup.find_all('div', class_='view_step_cont'):
#         step_desc = step_section.find('div', class_='media-body').get_text(strip=True) if step_section.find('div', 'media-body') else None
#         instructions.append(step_desc)

#     platform = extract_platform(url)

#     publishDate = None
#     publish_date_div = soup.find('p', class_='view_notice_date')
#     if publish_date_div:
#         publish_date_text = publish_date_div.find('b')
#         if publish_date_text and "등록일 :" in publish_date_text.get_text(strip=True):
#             publishDate = publish_date_text.get_text(strip=True).split("등록일 :")[1].strip()

#     author = soup.find('span', class_='user_info2_name').get_text(strip=True) if soup.find('span', 'user_info2_name') else None

#     imgUrl = soup.find('img', id='main_thumbs')['src'] if soup.find('img', id='main_thumbs') else None

#     tools = []
#     tools_section = soup.select('#contents_area_full > div.cont_ingre2 > div:nth-child(4) > ul > li')
#     if tools_section:
#         for item in tools_section:
#             name = item.select('div.ingre_list_name')
#             if name:
#                 tool_name = name[0].get_text(strip=True)
#                 tools.append(tool_name)

#     recipe = {
#         "title": title,
#         "oriUrl": url,
#         "serving": servings,
#         "cookingTime": cookingTime,
#         "level": level,
#         "ingredients": ingredients,
#         "instructions": instructions,
#         "platform": platform,
#         "publishDate": publishDate,
#         "createDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "imgUrl": imgUrl,
#         "author": author,
#         "tools": tools
#     }

#     result = recipes_collection.insert_one(recipe)
#     if result.inserted_id:
#         new_recipes_count[0] += 1
#         logging.info(f"Recipe {title} inserted with id {result.inserted_id}. Total recipes saved so far: {new_recipes_count[0]}")
#         return recipe
#     return False

####################