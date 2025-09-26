import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_PATH = "../data/chroma_db"
JSON_PATH = "../data/horse_diseases_docs.json"

def load_chroma(json_path=JSON_PATH, persist_directory=CHROMA_PATH):
    # 🔹 Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🔹 Convert setiap entry JSON ke Document
    docs = [
        Document(
            page_content=doc["document"],   # langsung isi dengan teks utamanya
            metadata={
                "id": doc["id"],
                "name": doc["metadata"]["name"],
                "category": doc["metadata"]["category"],
                "url": doc["metadata"]["url"]
            }
        )
        for doc in data
    ]

    # 🔹 Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 Buat vektor DB
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"Chroma DB created at {persist_directory} with {len(docs)} documents")

    return vectordb


if __name__ == "__main__":
    load_chroma()