import os
#추가
from batch_processor import batch_process_documents

def main():
    persist_directory = "./chroma_db"
    collection_metadata = {'hnsw:space': 'cosine'}
    model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        data_dir = "crawled_data"
        batch_size = 11  # 한번에 처리할 파일의 수
        batch_process_documents(data_dir, batch_size, model_name, persist_directory, collection_metadata)
        print("Batch processing complete. All documents processed and stored.")
    else:
        print("Database already exists.")

if __name__ == "__main__":
    main()