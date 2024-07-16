# database.py
from sqlalchemy import create_engine, Column, Integer, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql+psycopg2://iam:iam%40123@localhost:5432/iam"
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(LargeBinary, nullable=False)

def get_session():
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    try:
        Base.metadata.create_all(engine)
        print("Tables created or verified successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
