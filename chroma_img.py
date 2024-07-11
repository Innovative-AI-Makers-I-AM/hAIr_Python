### 이미지를 벡터화하여 Chroma DB 에 임베딩값 저장 ###
# import os
# from PIL import Image
# import numpy as np
# import torch
# from transformers import ViTImageProcessor, ViTModel
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

import os
from langchain_community.vectorstores import Chroma
from langchain_community.llms import openai
from langchain.chains import retrieval_qa
# OpenAI API 키 설정

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

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

model = SentenceTransformer('upskyy/kf-deberta-multitask')

# input 데이터
hairstyle_data = [
    {
      "style_name": "레이어드 컷",
      "gender": ["남", "여"],
      "hair_length": ["롱", "미디움"],
      "bangs": ["유", "무"],
      "face_shape": ["둥근형", "계란형", "긴형", "네모형", "다이아몬드형"],
      "description": "레이어드 컷은 여러 층으로 나누어져 있어 자연스러운 볼륨감과 움직임을 줍니다. 긴 머리와 중간 길이의 머리에 잘 어울리며, 다양한 얼굴형에 적합합니다.",
      "image_link": "https://contents-cdn.viewus.co.kr/image/2024/03/CP-2022-0277/image-f3905fb3-8f54-4e83-be13-444dfdb3acd3.jpeg"
    },
    {
      "style_name": "단발 컷",
      "gender": ["여"],
      "hair_length": ["단발"],
      "bangs": ["유"],
      "face_shape": ["계란형", "둥근형"],
      "description": "단발 컷은 깔끔하고 세련된 스타일로, 계란형과 둥근형 얼굴형에 잘 어울립니다.",
      "image_link": "https://example.com/images/bob_cut.jpg"
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
        "gender":', '.join(data['gender']),
        "hair_length":', '.join(data['hair_length']),
        "bangs":', '.join(data['bangs']),
        "face_shape":  ', '.join(data['face_shape']),
        "description": data['description'],
        "image_link": data['image_link']
    })

embeddings = model.encode(texts) # input 데이터 임베딩으로 변환
embeddings = [embedding.tolist() for embedding in embeddings] # 임베딩값 list 형식으로 받음


# 임베딩 데이터 추가
ids = [str(i+1) for i in range(len(embeddings))]
# metadatas = [{"source": f"text_{i}",
#                "author": f"author_name{i}",
#                 "category": f"example_category{i}"} for i in range(len(texts))]
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

# Embedding Function Wrapper
class EmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode([query])[0].tolist()
    
embedding_function = EmbeddingFunction(model)

# 검색기 설정
vectorstore = Chroma(collection_name='text_embeddings', embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})

# 질문하기
question = input('Ask your question: ')

# # 텍스트 임베딩 생성
# question_embedding = model.encode([question])[0]

# 검색 실행
retrieved_docs = retriever.get_relevant_documents(query=question)

# 결과 출력
print("Recommended hairstyles:")
if retrieved_docs:
    print('Recommended hairstyle : ')
    doc = retrieved_docs[0]
    print(f"Style name: {doc.metadata['style_name']}")
    print(f"Description: {doc.metadata['description']}")
    print(f"Image link: {doc.metadata['image_link']}")
else:
    print("No matching hairstyle found.")

# for doc in retrieved_docs:
#     print(f"Recommended style name: {doc.metadata['style_name']}")
#     print(f"Description: {doc.metadata['description']}")
#     print(f"Image link: {doc.metadata['image_link']}")
#     print('---')

# RetrievalQA 체인 설정
qa = retrieval_qa.from_chain_type(llm=openai(), chain_type="stuff", retriever=retriever, return_source_documents=True)

# 질문에 대한 답변 생성
result = qa({"query": question})
print(f'\nAnswer : {result["result"]}')



###########################################

# # ViTImageProcessor와 ViTModel 불러오기
# processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
# model = ViTModel.from_pretrained('facebook/dino-vits16')

# print('Models load')

# # 이미지 폴더 경로 설정
# image_folder = 'images'
# image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

# # 이미지 임베딩 생성 및 ChromaDB에 저장
# embeddings = []
# documents = []
# metadatas = []
# ids = []

# for idx, image_file in enumerate(image_files):
#     image_path = os.path.join(image_folder, image_file)
#     img = Image.open(image_path).convert('RGB')
    
#     # 이미지 텐서로 변환
#     img_tensor = processor(images=[img], return_tensors='pt')
#     outputs = model(**img_tensor)
    
#     # 임베딩 추출
#     embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().squeeze()
    
#     # 임베딩, 문서 및 메타데이터 준비
#     embeddings.append(embedding.tolist())
#     documents.append(image_path)
#     metadatas.append({"file_path": image_path})
#     ids.append(str(idx))

# # ChromaDB에 삽입
# collection.add(
#     documents=documents,
#     embeddings=embeddings,
#     metadatas=metadatas,
#     ids=ids
# )
# print("All images have been processed and stored in ChromaDB.")

# # ChromaDB에서 저장된 값 가져오기
# stored_data = collection.get(include=["documents", "embeddings", "metadatas"])

# # 저장된 데이터 출력
# for i in range(len(stored_data['ids'])):
#     print(f"ID: {stored_data['ids'][i]}")
#     print(f"Document: {stored_data['documents'][i]}")
#     print(f"Embedding (first 5 elements): {stored_data['embeddings'][i][:5]}")  # 임베딩의 처음 5개 요소만 출력
#     print(f"Metadata: {stored_data['metadatas'][i]}")
#     print("---")































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