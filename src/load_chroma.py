import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import shutil

from src.utils.model_config import MODEL_CONFIG

embedding_model_name = MODEL_CONFIG["embedding_model_name"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
JSON_PATH = os.path.join(DATA_DIR, "horse_diseases_docs.json")

def load_chroma(json_path=JSON_PATH, persist_directory=CHROMA_PATH):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Existing Chroma DB at {persist_directory} removed.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [
        Document(
            page_content=doc["document"], 
            metadata={
                "id": doc["id"],
                "name": doc["metadata"]["name"],
                "category": doc["metadata"]["category"],
                "url": doc["metadata"]["url"]
            }
        )
        for doc in data
    ]

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"Chroma DB created at {persist_directory} with {len(docs)} documents")

    return vectordb


if __name__ == "__main__":
    load_chroma()