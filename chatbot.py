from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI  # 올바른 경로로 가져오기
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# 초기 프롬프트 템플릿 설정 (한국어) 사용자가 처음 입력한 텍스트에 대해 응답하는 템플릿
initial_prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "당신은 전문 헤어 디자이너입니다. 사용자가 미용실에 있는 것처럼 대화를 나누세요. "
        "사용자의 질문에만 답변하고, 전문 헤어 디자이너처럼 대화 해주세요. "
        "사용자: {input_text}\n"
        "전문 헤어 디자이너:"
    )
)

# 대화 지속을 위한 프롬프트 템플릿 대화가 계속될 때, 이전 대화 내용을 포함하여 응답하는 템플릿
ongoing_prompt_template = PromptTemplate(
    input_variables=["conversation_history", "input_text"],
    template=(
        "당신은 전문 헤어 디자이너입니다. 다음은 사용자와의 대화 내용입니다:\n"
        "{conversation_history}\n"
        "사용자: {input_text}\n"
        "전문 헤어 디자이너:"
    )
)

# 요약을 위한 프롬프트 템플릿
summary_prompt_template = PromptTemplate(
    input_variables=["conversation_history"],
    template=(
        "당신은 전문 헤어 디자이너입니다. 다음은 사용자와의 대화 내용입니다:\n"
        "{conversation_history}\n"
        "사용자가 원하는 머리스타일을 간단히 요약해 주세요."
    )
)

# LLMChain 설정
llm_chain = LLMChain(
    llm=OpenAI(api_key=openai_api_key, max_tokens=150, temperature=0.7, top_p=0.9),
    prompt=initial_prompt_template
)

# 대화 이력을 저장하기 위한 conversation_history 리스트 초기화
conversation_history = []

def chat():
    initial_message = "안녕하세요! 어떤 머리스타일을 원하시나요?"
    print(f"전문 헤어 디자이너: {initial_message}")
    conversation_history.append(f"전문 헤어 디자이너: {initial_message}")
    
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["종료", "끝"]:
            break
        
        conversation_history.append(f"사용자: {user_input}")
        
        # 대화 이력을 문자열로 결합
        conversation_text = "\n".join(conversation_history)
        
        # LLMChain 실행
        ongoing_chain = LLMChain(
            llm=OpenAI(api_key=openai_api_key, max_tokens=150, temperature=0.7, top_p=0.9),
            prompt=ongoing_prompt_template
        )
        response = ongoing_chain.run(conversation_history=conversation_text, input_text=user_input)
        
        conversation_history.append(f"전문 헤어 디자이너: {response.strip()}")
        print(f"전문 헤어 디자이너: {response.strip()}")
    
    # 요약 프롬프트 실행
    summary_chain = LLMChain(
        llm=OpenAI(api_key=openai_api_key, max_tokens=150, temperature=0.7, top_p=0.9),
        prompt=summary_prompt_template
    )
    conversation_text = "\n".join(conversation_history)
    summary_response = summary_chain.run(conversation_history=conversation_text)
    
    print(f"\n대화 요약: {summary_response.strip()}")

if __name__ == "__main__":
    chat()
