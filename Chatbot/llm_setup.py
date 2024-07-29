from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_transformers import LongContextReorder
from typing import List, Dict, Any

#추가
from langchain.chains import ConversationalRetrievalChain

def setup_llm_and_retrieval_qa(db, model_name, temperature, max_tokens, prompt_template):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")

    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\n{question}")
    ])

    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})

    # Document Transformer 적용
    document_transformer = LongContextReorder()
    # retriever.add_document_transformer(document_transformer)

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    #     memory=memory,
    #     chain_type_kwargs={"prompt": chat_prompt},
    #     # document_transformers=[document_transformer]  # Transformer를 여기서 추가
    # )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        return_source_documents=True,
        return_generated_question=True,
        output_key="result"  # 이 줄을 추가
    )

    return qa


#----------------------------------- 필요 없는 코드------------------------------------------

# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain_community.document_transformers import LongContextReorder

# # 새로 추가된 함수: 검색된 문서를 변환하는 함수
# def transform_documents(documents, transformer):
#     return [transformer.transform(document) for document in documents]

# async def setup_llm_and_retrieval_qa(db, model_name, temperature, max_tokens, prompt_template):
#     llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")

#     system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
#     chat_prompt = ChatPromptTemplate.from_messages([
#         system_message_prompt,
#         HumanMessagePromptTemplate.from_template("Context:\n{context}\n\n{question}")
#     ])

#     retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 12})

#     # Document Transformer 적용
#     document_transformer = LongContextReorder()

#     # 수정된 부분: 문서 변환을 포함한 QA 파이프라인 함수 정의
#     async def qa_pipeline(query):
#         docs = await retriever.get_relevant_documents(query["query"])  # 수정된 부분: 적절한 검색 메서드 사용
#         transformed_docs = transform_documents(docs, document_transformer)
#         response = await llm(
#             messages=[
#                 {"role": "system", "content": system_message_prompt.prompt},
#                 {"role": "user", "content": query["query"]},
#                 {"role": "assistant", "content": chat_prompt.format(context=transformed_docs, question=query["query"])}
#             ]
#         )
#         return {"result": response["choices"][0]["message"]["content"], "sources": transformed_docs}

#     return qa_pipeline