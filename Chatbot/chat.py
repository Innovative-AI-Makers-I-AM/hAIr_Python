def run_chatbot(qa):
    while True:
        query = input("질문: ")
        if query in ["exit", "quit"]:
            break
        for token in qa.stream({"query": query}):
            print(token['result'], end='', flush=True)
        print()
