from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# Selenium WebDriver 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 브라우저 창을 띄우지 않음
driver = webdriver.Chrome()

# 스타일 목록
styles = ["레이어드컷", "허쉬컷", "샤기컷", "리프컷", "원랭스컷", "픽시컷", "보브컷", "뱅헤어", "비대칭컷", "머쉬룸컷"]

# 결과를 저장할 리스트
results = []

# 각 스타일에 대해 크롤링
for style in styles:
    try:
        # 웹페이지 URL
        url = f"https://hairshop.kakao.com/search/style?category=CUT&gender=FEMALE&tags={style}"
        driver.get(url)
        time.sleep(2)  # 페이지 로딩 대기

        # 첫 번째 이미지 URL 추출
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        first_image = soup.select_one('ul.list_likeStyleImageList img.img_g')
        image_url = first_image['src'] if first_image else None

        # 첫 번째 이미지 클릭
        first_image_element = driver.find_element(By.CSS_SELECTOR, 'ul.list_likeStyleImageList img.img_g')
        first_image_element.click()
        time.sleep(2)  # 페이지 로딩 대기

        # 해시태그 추출 (최대 3개)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        hashtags = soup.select('div.cover_tag a.link_hashtag')
        hashtag_texts = [tag.text for tag in hashtags[:3]]  # 최대 3개만 추출

        results.append({
            'style': style,
            'image_url': image_url,
            'hashtags': ', '.join(hashtag_texts)
        })
    except Exception as e:
        print(f"Error processing style {style}: {e}")

# WebDriver 종료
driver.quit()

# 결과를 CSV 파일로 저장
df = pd.DataFrame(results)
df.to_csv('hairstyles.csv', index=False)

print("크롤링 완료 및 CSV 파일 저장 완료.")