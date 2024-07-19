# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup
# import pandas as pd
# import time

# # Selenium WebDriver 설정
# options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # 브라우저 창을 띄우지 않음
# driver = webdriver.Chrome()

# # 스타일 목록
# styles = ["레이어드컷", "허쉬컷", "샤기컷", "리프컷", "원랭스컷", "픽시컷", "보브컷", "뱅헤어", "비대칭컷", "머쉬룸컷"]

# # 결과를 저장할 리스트
# results = []

# # 각 스타일에 대해 크롤링
# for style in styles:
#     try:
#         # 웹페이지 URL
#         url = f"https://hairshop.kakao.com/search/style?category=CUT&gender=FEMALE&tags={style}"
#         driver.get(url)
#         time.sleep(2)  # 페이지 로딩 대기

#         # 첫 번째 이미지 URL 추출
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         first_image = soup.select_one('ul.list_likeStyleImageList img.img_g')
#         image_url = first_image['src'] if first_image else None

#         # 첫 번째 이미지 클릭
#         first_image_element = driver.find_element(By.CSS_SELECTOR, 'ul.list_likeStyleImageList img.img_g')
#         first_image_element.click()
#         time.sleep(2)  # 페이지 로딩 대기

#         # 해시태그 추출 (최대 3개)
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         hashtags = soup.select('div.cover_tag a.link_hashtag')
#         hashtag_texts = [tag.text for tag in hashtags[:3]]  # 최대 3개만 추출

#         results.append({
#             'style': style,
#             'image_url': image_url,
#             'hashtags': ', '.join(hashtag_texts)
#         })
#     except Exception as e:
#         print(f"Error processing style {style}: {e}")

# # WebDriver 종료
# driver.quit()

# # 결과를 CSV 파일로 저장
# df = pd.DataFrame(results)
# df.to_csv('hairstyles.csv', index=False)

# print("크롤링 완료 및 CSV 파일 저장 완료.")




# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# from docx import Document
# import json

# # 크롤링할 URL
# urls = ['https://hairshop.kakao.com/magazines/543', 'https://hairshop.kakao.com/magazines/545']

# # 전체 결과를 저장할 리스트
# all_data = []

# # 각 URL에 대해 크롤링 수행
# for i, url in enumerate(urls):
#     # HTTP GET 요청을 보내고 응답을 받음
#     response = requests.get(url)
#     response.raise_for_status()  # 요청이 성공했는지 확인

#     # BeautifulSoup을 사용하여 HTML 파싱
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # 크롤링할 태그 목록
#     tags = ['p', 'h1', 'h3', 'h4', 'h5', 'i', 'span', 'strong']

#     # 결과를 저장할 리스트
#     data = []

#     # 특정 텍스트와 태그를 제외하고 HTML의 순서를 유지하여 텍스트를 추출
#     for element in soup.find_all(text=True):
#         parent_tag = element.parent.name
#         text = element.strip()
#         if text and parent_tag in tags and text not in ["🔎 이 머리 어디서 했는지 궁금하다면? 사진을 눌러보세요!", "👩🏻‍🦱"] and parent_tag != 'h2':
#             data.append(text)

#     # 텍스트들을 하나의 문자열로 결합
#     full_text = ' '.join(data)
    
#     # 각 페이지의 데이터를 딕셔너리 형태로 저장
#     page_data = {
#         'url': url,
#         'content': full_text
#     }
    
#     # 전체 결과 리스트에 추가
#     all_data.append(page_data)

# # JSON 파일로 저장
# with open('crawled_data.json', 'w', encoding='utf-8') as f:
#     json.dump(all_data, f, ensure_ascii=False, indent=4)

# print("크롤링이 완료되었습니다. JSON 파일이 저장되었습니다.")

# # docx 파일 크롤링 로직
# # 각 URL에 대해 크롤링 수행
# for i, url in enumerate(urls):
#     # HTTP GET 요청을 보내고 응답을 받음
#     response = requests.get(url)
#     response.raise_for_status()  # 요청이 성공했는지 확인

#     # BeautifulSoup을 사용하여 HTML 파싱
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # 크롤링할 태그 목록
#     tags = ['p', 'h1', 'h3', 'h4', 'h5', 'i', 'span', 'strong']

#     # 결과를 저장할 리스트
#     data = []

#     # 특정 텍스트와 태그를 제외하고 HTML의 순서를 유지하여 텍스트를 추출
#     for element in soup.find_all(text=True):
#         parent_tag = element.parent.name
#         text = element.strip()
#         if text and parent_tag in tags and text not in ["🔎 이 머리 어디서 했는지 궁금하다면? 사진을 눌러보세요!", "👩🏻‍🦱"] and parent_tag != 'h2':
#             data.append(text)

#     # # 텍스트들을 하나의 문자열로 결합
#     # full_text = ' '.join(data)
    
#     # Document 객체 생성
#     doc = Document()
#     doc.add_paragraph(data)
    
#     # docx 파일로 저장
#     file_name = f'doc_{i+1}.docx'
#     doc.save(file_name)

# print("크롤링이 완료되었습니다. docx 파일들이 저장되었습니다.")


from chromadb import PersistentClient
client = PersistentClient()
collections = client.list_collections()
collection_names = [coll.name for coll in collections]
print(f'1. collection list :  {[collection_names]}')
collection_name = 'test_embeddings'
client.delete_collection(name=collection_name)
print(f'기존 컬렉션 {collection_name}을 삭제했습니다.')
collections = client.list_collections()
collection_names = [coll.name for coll in collections]
print(f'2. collection list :  {[collection_names]}')