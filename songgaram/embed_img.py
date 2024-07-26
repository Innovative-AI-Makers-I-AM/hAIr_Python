import numpy as np
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Initialize MTCNN and InceptionResnetV1 models
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to generate embedding for an image
def get_face_embedding(image_path):
    try:
        img = Image.open(image_path)
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            img_embedding = resnet(img_cropped.unsqueeze(0)).detach().numpy().flatten()
            return img_embedding
        else:
            return None
    except (FileNotFoundError, UnidentifiedImageError):
        print(f"File not found or cannot be identified: {image_path}")
        return None

# Read CSV file
csv_file = 'songgaram/hair_image/new_hairstyles_filtered.csv'
df = pd.read_csv(csv_file)

# Create a list to store texts, metadata, and embeddings
texts = []
metadatas = []
embeddings = []

for _, row in df.iterrows():
    image_path = f"images/{row['image_id']}.jpg"
    embedding = get_face_embedding(image_path)
    if embedding is not None:
        metadata = {
            'image_id': row['image_id'],
            'image_path': image_path,
            'sex': row['sex'],
            'category': row['category'],
            'length': row['length'],
            'style': row['style'],
            'designer': row['designer'],
            'shop_name': row['shop_name'],
            'hashtag1': row['hashtag1'],
            'hashtag2': row['hashtag2'],
            'hashtag3': row['hashtag3']
        }
        texts.append("")  # Placeholder text as Chroma expects some text input
        metadatas.append(metadata)
        embeddings.append(embedding.tolist())
        print(f"Embedding for {image_path}: {embedding}")  # Print the embedding
        print(metadatas)

# Initialize Chroma DB
collection_name = 'hair_image'
persist_directory = 'chroma_db'

# Create Chroma instance
chroma = Chroma(collection_name=collection_name, persist_directory=persist_directory)

# Manually add the embeddings and metadata to the collection
for i, embedding in enumerate(embeddings):
    chroma._collection.upsert(
        embeddings=[embedding],
        documents=[texts[i]],
        ids=[metadatas[i]['image_id']],
        metadatas=[metadatas[i]]
    )

# Retrieve all documents in the collection
def retrieve_all_embeddings():
    all_results = chroma._collection.get(include=['embeddings', 'metadatas'])
    return all_results

# Example usage: Retrieve all embeddings
all_results = retrieve_all_embeddings()
formatted_results = [
    {
        'id': all_results['ids'][i],
        'embedding': all_results['embeddings'][i][:5],
        'metadata': all_results['metadatas'][i]
    }
    for i in range(len(all_results['ids']))
]

for result in formatted_results:
    print(f"id: {result['id']}\nembedding: {result['embedding']}\nmetadata: {result['metadata']}\n")