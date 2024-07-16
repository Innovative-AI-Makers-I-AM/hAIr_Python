from sqlalchemy import create_engine, Column, Integer, LargeBinary
from sqlalchemy.orm import declarative_base

# 데이터베이스 설정
DATABASE_URL = "postgresql+psycopg2://iam:iam%40123@localhost:5432/iam"
engine = create_engine(DATABASE_URL, echo=True)

Base = declarative_base()
class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(LargeBinary, nullable=False)