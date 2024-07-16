from sqlalchemy import create_engine, Column, Integer, LargeBinary, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "postgresql+psycopg2://iam:iam%40123@localhost:5432/iam"
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(LargeBinary, nullable=False)
    meta = relationship("ImageMetadata", back_populates="image", uselist=False)

class ImageMetadata(Base):
    __tablename__ = 'image_metadata'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    sex = Column(String, nullable=False)
    length = Column(String, nullable=False)
    style = Column(String, nullable=False)
    designer = Column(String, nullable=False)
    shop_name = Column(String, nullable=False)
    hashtag1 = Column(String, nullable=False)
    hashtag2 = Column(String, nullable=False)
    hashtag3 = Column(String, nullable=False)
    image = relationship("Image", back_populates="meta")

def get_session():
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    try:
        Base.metadata.create_all(engine)
        print("Tables created or verified successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
