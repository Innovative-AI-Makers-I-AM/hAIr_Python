# 사용자 질문을 입력 받고, 챗봇의 `stream` 메서드를 통해 대답을 출력
def run_chatbot(qa):
    while True:
        query = input("질문: ")
        if query in ["exit", "quit"]:
            break
        for token in qa.stream({"query": query}):
            print(token['result'], end='', flush=True)
        print()