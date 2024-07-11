import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# OpenAI API 키 설정

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# 이미지 파일 경로 설정
image_dir = 'images'

# CLIP 모델 및 프로세서 설정
model_name = 'upskyy/kf-deberta-multitask'
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 이미지 파일 경로 리스트로 가져오기
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]

# Chroma 벡터 스토어 생성
vectorstore = Chroma(collection_name='hairstyle_image', embedding_function=None)

for image_path in image_paths:
  # 이미지 로드
  image = Image.open(image_path)
  inputs = processor(images=image, return_tensors='pt')

  # 파일명에서 hairstyle 추출
  hairstyle = os.path.basename(image_path)

  # 이미지 임베딩 생성
  with torch.no_grad():
      image_embedding = model.get_image_features(**inputs).numpy().flatten()

  # 메타데이터 생성
  image_metadata = {
      'image_path': image_path,
      'hairstyle': hairstyle,
      'gender': 'female'
  }

  # Chroma에 이미지 임베딩과 메타데이터 저장
  vectorstore.add_documents(documents=None, metadatas=[image_metadata], embeddings=[image_embedding])

# 텍스트 임베딩 함수 설정
text_embeddings = OpenAIEmbeddings()

# 검색기 설정
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# 질문하기
question = input('Ask your question: ')

# 텍스트 임베딩 생성
text_embedding = text_embeddings.embed_query(question)

# 검색 실행
retrieved_docs = retriever.get_relevant_documents(query=text_embedding)

# 결과 출력
print("Recommended images:")
for doc in retrieved_docs:
  print(f"Recommended image path: {doc.metadata['image_path']}")
  image = Image.open(doc.metadata['image_path'])
  image.show()

# RetrievalQA 체인 설정
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

# 질문에 대한 답변 생성
result = qa({"query": question})
print(result['result'])

#------------------------#


# import os

# from langchain.chains import retrieval_qa
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.llms import openai
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnableMap
# from langchain.schema import Document

# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel
# import torch.nn.functional as F

# # OpenAI API 키 설정
# # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# # 이미지 파일 경로 설정
# image_dir = 'images'

# # CLIP 모델 및 프로세서 설정
# model_name = 'openai/clip-vit-base-patch32'
# model = CLIPModel.from_pretrained(model_name)
# processor = CLIPProcessor.from_pretrained(model_name)

# # 이미지 파일 경로 리스트로 가져오기
# image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]

# # Chroma 벡터 스토어 생성
# vectorstore = Chroma(collection_name='hairstyle_image', embedding_function=None)

# # 이미지 임베딩 모델 설정
# # documents = []
# # embeddings = []
# for image_path in image_paths:
#   # 이미지 로드
#   image = Image.open(image_path)
#   inputs = processor(images=image, return_tensors='pt')

#   # 파일명에서 hairstyle 추출
#   hairstyle = os.path.basename(image_path)

#   # 이미지 임베딩 생성
#   with torch.no_grad():
#     image_embedding = model.get_image_features(**inputs).numpy().flatten()
  
#   # 메타데이터 생성
#   image_metadata = {
#      'image_path' : image_path,
#      'hairstyle' : hairstyle,
#      'gender' : 'female'
#   }

#   # Chroma에 이미지 임베딩과 메타데이터 저장
#   vectorstore.add_documents(documents=None, metadatas=[image_metadata], embeddings=[image_embedding])

#   # # Document 생성(추가)
#   # doc = Document(page_content="Image description placeholder", metadata=image_metadata)
#   # documents.append(doc)
#   # embeddings.append(embedding)

#   # # image_embeddings.append(embedding)
#   # # metadata.append(image_metadata)

# # # Chroma 벡터 스토어 생성
# # vectorstore = Chroma(collection_name='hairstyle_image')

# # vectorstore.add_texts(
# #    texts=[doc.page_content for doc in documents],
# #    metadatas=[doc.metadata for doc in documents],
# #    embeddings=embeddings
# # )

# question = input('Ask your question:')

# # 텍스트 임베딩 함수 설정
# text_inputs = processor(text=question, return_tensors='pt')
# with torch.no_grad():
#    text_embedding = model.get_text_features(**text_inputs).numpy().flatten()


# # 검색기 설정
# retriever = vectorstore.as_retriever(
#    search_type = 'similarity',
#    search_kwargs={'k':3, 'fetch_k':10}
# )

# # # 질문하기
# # question = input('Ask your question:')

# # 질문 텍스트 임베딩 생성
# text_inputs = processor(text=question, return_tensors='pt')
# with torch.no_grad():
#    text_embedding = model.get_text_features(**text_inputs).numpy().flatten()

# # 검색 실행
# retrieved_docs = retriever.get_relevant_documents(query={'embedding': text_embedding})

# # 결과 출력
# print("Recommended images:")
# for doc in retrieved_docs:
#     print(f"Recommended image path: {doc.metadata['image_path']}")
#     image = Image.open(doc.metadata['image_path'])
#     image.show()

# # ChatGPT를 사용하여 추천 이미지에 대한 설명 생성
# chatgpt = ChatOpenAI(model_name='gpt-4', temperature=0)
# qa = retrieval_qa(
#     llm=chatgpt,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True
# )
# result = qa({"query": question})
# print(result['result'])

#-----------------------------------------

# # Chroma 벡터 스토어 생성
# docsearch = Chroma.embeddings(image_embeddings, metadata)

# # 검색기 설정
# retriever = docsearch.as_retriever(
#   search_type='similarity',
#   search_kwrgs={'k':3, 'fetch_k' :10}
# )

# # 질문하기
# question = input('Ask your question:')

# # 질문 텍스트 임베딩 생성
# text_inputs = processor(text=question, return_tensors='pt')
# with torch.no_grad():
#    text_embedding = model.get_text_features(**text_inputs).numpy().flatten()

# # 임베딩 정규화
# image_embeds = F.normalize(image_embeddings, p=2, dim=1)
# text_embeds = F.normalize(text_embedding, p=2, dim=1)

# # 유사도 계산
# similarities = torch.matmul(text_embeds, image_embeds.T).squeeze().tolist()
# print(similarities)

# # 가장 유사한 이미지 추천
# top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
# recommended_images = [image_paths[i] for i in top_indices]

# # 결과 출력
# for image_path in recommended_images:
#     image = Image.open(image_path)
#     image.show()

# # ChatGPT를 사용하여 추천 이미지에 대한 설명 생성
# qa = retrieval_qa.from_chain_type(
#     llm=ChatOpenAI(model_name='gpt-4'),
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True
# )
# result = qa({"query": question})
# print(result['result'])



# # 프롬프트 템플릿 설정
# template = """Based on the following question, recommend the most relevant hairstyle image:
# {context}
# Question : {question}"""
# prompt = ChatPromptTemplate.from_template(template)

# # ChatGPT 모델 설정
# chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)

# # 체인 설정
# chain = RunnableMap({
#   'context' : lambda x: retriever.get_relevant_documents(x['question']),
#   'question': lambda x: x['question']
# }) | prompt | chatgpt


# # 검색기 실행
# relevant_docs = retriever.get_relevant_documents({'embedding': text_embeds})



# # 결과 출력
# recommended_images = [doc.metadata['image_path'] for doc in relevant_docs]
# for image_path in recommended_images:
#     image = Image.open(image_path)
#     image.show()




###############################################################################




# import os
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnableMap

# # OpenAI API 키 설정
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# # PDF 파일 경로 설정
# pdf_path = "path/to/your/pdf/file.pdf"

# # PDF 로드 및 분할
# loader = PyPDFLoader(pdf_path)
# pages = loader.load_and_split()

# # 텍스트 분할
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# texts = text_splitter.split_documents(pages)

# # OpenAI 임베딩 모델 설정
# embeddings = OpenAIEmbeddings()

# # Chroma 벡터 스토어 생성
# docsearch = Chroma.from_documents(texts, embeddings)

# # 검색기 설정
# retriever = docsearch.as_retriever(
#     search_type="mmr",
#     search_kwargs={'k': 3, 'fetch_k': 10}
# )

# # 프롬프트 템플릿 설정
# template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # ChatGPT 모델 설정
# chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # 체인 설정
# chain = RunnableMap({
#     "context": lambda x: retriever.get_relevant_documents(x['question']),
#     "question": lambda x: x['question']
# }) | prompt | chatgpt

# # 질문하기
# question = "혁신성장 정책금융에 대해서 설명해줘"
# result = chain.invoke({'question': question})

# # 결과 출력
# print(result.content)