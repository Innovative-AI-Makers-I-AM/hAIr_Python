from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from modules import Image  # Image 클래스를 정의한 모듈

# 데이터베이스 설정
DATABASE_URL = "postgresql+psycopg2://iam:iam%40123@localhost:5432/iam"
engine = create_engine(DATABASE_URL, echo=True)

Base = declarative_base()

# 이미지 테이블 정의
# class Image(Base):
#     __tablename__ = 'images'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     data = Column(LargeBinary, nullable=False)

# 테이블 생성
try:
    Base.metadata.create_all(engine)
    print("Tables created or verified successfully.")
except Exception as e:
    print(f"Error creating tables: {e}")

Session = sessionmaker(bind=engine)
session = Session()

print("temp_save_image_postgre 실행")

# 이미지 저장 함수
def save_image_to_db(file_path):
    try:
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            new_image = Image(data=binary_data)
            session.add(new_image)
            session.commit()
            print(f"Image saved with ID: {new_image.id}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error saving image to DB: {e}")

# 이미지 파일 경로
image_path = 'F:/Repository/hAIr_Python/images/Autumn Gaze.jpg'  # 실제 경로로 변경하세요

# 이미지 저장 호출
save_image_to_db(image_path)
