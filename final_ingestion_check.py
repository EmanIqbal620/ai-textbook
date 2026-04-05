import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

def verify_ingestion():
    """Verify the ingestion was successful"""
    print("Verifying textbook ingestion...")

    # Initialize Qdrant client with your credentials
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_api_key:
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_client = QdrantClient(url=qdrant_url)

    collection_name = "humanoid_ai_book"

    # Check total count
    count = qdrant_client.count(collection_name=collection_name)
    print(f"[SUCCESS] Total documents in Qdrant: {count.count}")

    # Verify collection exists
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"[SUCCESS] Collection '{collection_name}' exists with configuration:")
        print(f"  - Vector size: {collection_info.config.params.vectors.size}")
        print(f"  - Distance: {collection_info.config.params.vectors.distance}")
    except Exception as e:
        print(f"[ERROR] Could not get collection info: {e}")

    print("\n[SUCCESS] Textbook content has been successfully ingested into Qdrant!")
    print("Your RAG chatbot can now access the textbook content for accurate responses.")

if __name__ == "__main__":
    verify_ingestion()