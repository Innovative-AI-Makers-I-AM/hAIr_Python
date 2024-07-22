from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

def setup_llm_and_retrieval_qa(db, model_name, temperature, max_tokens, prompt_template):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")

    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\n{question}")
    ])

    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 5})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        chain_type_kwargs={"prompt": chat_prompt}
    )
    return qa
