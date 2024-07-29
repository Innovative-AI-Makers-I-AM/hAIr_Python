from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_transformers import LongContextReorder
from typing import List, Dict, Any

# ConversationalRetrievalChain을 사용하기 위해 추가
from langchain.chains import ConversationalRetrievalChain

def setup_llm_and_retrieval_qa(db, model_name, temperature, max_tokens, prompt_template):
    # ChatOpenAI 인스턴스를 생성하여 언어 모델을 설정
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    
    # 대화의 문맥을 기억하는 메모리 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")

    # 시스템 메시지 프롬프트를 설정
    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
    
    # 대화 프롬프트 템플릿을 설정, 사용자 입력을 포함
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\n{question}")
    ])

    # 데이터베이스에서 문서를 검색하기 위한 retriever 설정, MMR (Maximal Marginal Relevance) 알고리즘을 사용
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})

    # ConversationalRetrievalChain을 사용하여 대화형 검색 체인을 설정
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,  # 언어 모델 설정
        retriever=retriever,  # 문서 검색기 설정
        memory=memory,  # 대화 문맥을 기억하는 메모리 설정
        combine_docs_chain_kwargs={"prompt": chat_prompt},  # 문서를 결합할 때 사용할 프롬프트 설정
        return_source_documents=True,   # 소스 문서를 반환하도록 설정
        return_generated_question=True, # 생성된 질문을 반환하도록 설정
        # 이 줄을 추가
        output_key="result"   # 결과를 반환할 키 설정
    )

    return qa
