import json
from chromadb import Client
from chromadb.config import Settings

def load_json_to_chroma(json_path="data/horse_diseases_docs.json", persist_dir="data/chroma_db"):
    client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))

    if "horse_diseases" in [c.name for c in client.list_collections()]:
        client.delete_collection("horse_diseases")

    collection = client.create_collection("horse_diseases")

    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    collection.add(
        documents=[d["document"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
        ids=[d["id"] for d in docs]
    )

    print(f"oaded {len(docs)} documents into ChromaDB at {persist_dir}")
    return collection

if __name__ == "__main__":
    load_json_to_chroma()
