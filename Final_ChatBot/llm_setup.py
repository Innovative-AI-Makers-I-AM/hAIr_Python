from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_transformers import LongContextReorder
from typing import List, Dict, Any

# Langchain 라이브러리를 사용하여 LLM(Large Language Model) 기반의 질의응답 시스템을 설정하는 함수를 제공
def setup_llm_and_retrieval_qa(db, model_name, temperature, max_tokens, prompt_template):

    # ChatOpenAI를 사용하여 OpenAI의 Chat 모델을 로딩
    llm = ChatOpenAI(model_name=model_name, 
                     temperature=temperature, 
                     max_tokens=max_tokens, 
                     streaming=True, 
                     callbacks=[StreamingStdOutCallbackHandler()])
    
    # ConversationBufferMemory를 사용하여 대화 내역을 저장
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")

    # 매개변수로 prompt_template을 받아서 기본 프롬프트 생성
    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
    
    # ChatPromptTemplate을 사용하여 시스템 메시지와 사용자 메시지를 정의
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\n{question}")
    ])

    # 벡터 데이터베이스에서 관련 문서를 검색하는 검색기를 생성
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 15})

    # Document Transformer 적용
    document_transformer = LongContextReorder()
    # retriever.add_document_transformer(document_transformer)

    # 검색된 문서와 LLM을 결합한 질의응답 체인을 생성
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        chain_type_kwargs={"prompt": chat_prompt},
        # document_transformers=[document_transformer]  # Transformer를 여기서 추가
    )

    return qa