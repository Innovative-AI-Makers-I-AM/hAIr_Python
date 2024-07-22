def run_chatbot(qa):
    while True:
        query = input("질문: ")
        if query in ["exit", "quit"]:
            break
        for token in qa.stream({"query": query}):
            print(token['result'], end='', flush=True)
<<<<<<< HEAD
        print()
=======
        print()
>>>>>>> b4f7cf9ac16963340719684ad07c7555c71a65b4
