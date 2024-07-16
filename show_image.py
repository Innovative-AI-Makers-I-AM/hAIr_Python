import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from modules import Image  # Image 클래스를 정의한 모듈

print("import 끝!")
# 데이터베이스 설정
DATABASE_URL = "postgresql+psycopg2://iam:iam%40123@localhost:5432/iam"
print("Database 설정!")
engine = create_engine(DATABASE_URL)
print("Engine 설정!")
Session = sessionmaker(bind=engine)
print("Session 설정!")
session = Session()
print("Session 실행!")

print("show image 설정!")
def show_image(image_id):
    try:
        # 이미지 데이터베이스에서 읽기
        image = session.query(Image).filter(Image.id == image_id).first()
        if image is not None:
            print("이미지 있습니다.!")
            # 이미지 데이터를 numpy 배열로 변환
            nparr = np.frombuffer(image.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 이미지 보여주기
            cv2.imshow(f'Image ID: {image_id}', img)
            cv2.waitKey(0)  # 키 입력 대기
            cv2.destroyAllWindows()  # 모든 창 닫기
        else:
            print(f"이미지 없는데요?")
            print(f"No image found with ID {image_id}")
    except Exception as e:
        print(f"Error retrieving image from DB: {e}")
print("show image 끝!")

print("list_images 설정!")
def list_images():
    try:
        images = session.query(Image).all()
        for image in images:
            print(f"Image ID: {image.id}, Data length: {len(image.data)}")
    except Exception as e:
        print(f"Error listing images: {e}")
print("list_images 끝!")

# 데이터베이스에 저장된 이미지 확인
list_images()

# 예제: ID가 1인 이미지 보여주기 (존재하지 않는 ID를 사용하여 테스트)
show_image(11)