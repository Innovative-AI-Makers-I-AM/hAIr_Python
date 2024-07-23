## chat.py
**사용자 입력을 받아 챗봇과 상호작용하는 간단한 루프를 구현.**
- run_chatbot 함수는 무한 루프를 돌며 사용자의 입력을 받아 처리

## chatbot.py
**.env 파일을 로드하여 OpenAI API 키를 설정하고 데이터베이스를 로드 및 생성하여 챗봇을 실행.**

## data_processing.py
**텍스트 파일을 로드하고 텍스트를 분할하는 기능을 제공.**
- load_and_split_documents 함수는 데이터 디렉토리의 텍스트 파일들을 로드하고, 이를 작은 청크로 나눕니다.

## llm_setup.py
**LLM 및 검색 기반 QA 시스템을 설정.**
- setup_llm_and_retrieval_qa 함수는 모델을 설정하고, 검색 및 QA 시스템을 초기화

## vector_store.py
**문서를 벡터화하여 저장하거나 기존 벡터 스토어를 로드하는 기능을 제공.**
- create_vector_store 함수는 문서를 벡터화하고 데이터베이스에 저장
- load_vector_store 함수는 기존 벡터 스토어를 로드