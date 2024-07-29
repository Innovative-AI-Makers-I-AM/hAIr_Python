# 전체 파일 구조 및 설명
## api.py
FastAPI를 사용하여 챗봇 API를 구현한 파일입니다.

## batch_processor.py
문서들을 배치로 처리하여 ChromaDB에 저장하는 파일입니다.

## chat.py (현재 안 씀)
챗봇을 콘솔에서 실행할 수 있는 파일입니다.

## chroma_db_utils.py
Chroma DB를 사용하여 임베딩 데이터를 저장, 관리, 검색하는 기능을 제공  
얼굴 추천 기능에서 사용

## data_processing.py (현재 안 씀)
문서를 로드하고 분할하는 기능을 제공하는 파일입니다.

## llm_setup.py
Langchain과 ChromaDB를 설정하여 질의응답 시스템을 구성하는 파일입니다.

## main.py
챗봇을 위한 벡터 데이터베이스를 생성하는 메인 스크립트

## vector_store.py
ChromaDB와 관련된 벡터 저장소를 생성하고 로드하는 파일입니다.

# 주요 파일의 실행 순서
1. 데이터 처리 및 데이터베이스 초기화먼저, main.py 파일을 실행하여 데이터베이스를 초기화합니다. 이 단계는 크롤링된 데이터(crawled_data 디렉토리에 있는 txt 파일들)를 처리하고 ChromaDB에 저장합니다.

2. FastAPI 서버 실행
api.py 파일을 실행하여 FastAPI 서버를 시작합니다. 이 서버는 사용자로부터 메시지를 받고, ChromaDB를 이용하여 적절한 답변을 생성합니다.

3. 콘솔에서 챗봇 실행 (선택 사항)
chat.py 파일을 실행하여 콘솔에서 직접 챗봇과 상호작용할 수 있습니다.