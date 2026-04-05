import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="backend/.env")

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def run_sitemap_ingestion():
    """Run the sitemap ingestion to ensure all textbook content is in Qdrant"""
    print("Running sitemap ingestion to populate Qdrant with textbook content...")

    # Import the sitemap ingestion class
    from backend.ingestion.sitemap_ingestion import SitemapIngestion

    # Get environment variables
    sitemap_url = os.getenv("BOOK_BASE_URL", "https://humanoid-robotics-textbook-4ufa.vercel.app/")
    sitemap_xml_url = f"{sitemap_url.rstrip('/')}/sitemap.xml"

    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    print(f"Sitemap URL: {sitemap_xml_url}")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection: humanoid_ai_book")

    if not cohere_api_key:
        print("[ERROR] COHERE_API_KEY not found in environment")
        return False

    if not qdrant_url:
        print("[ERROR] QDRANT_URL not found in environment")
        return False

    try:
        # Create ingestion instance
        ingestion = SitemapIngestion(
            sitemap_url=sitemap_xml_url,
            cohere_api_key=cohere_api_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )

        print("Starting sitemap ingestion...")
        ingestion.run_ingestion()

        print("[SUCCESS] Sitemap ingestion completed!")
        return True

    except Exception as e:
        print(f"[ERROR] Sitemap ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_current_content():
    """Check how many documents are currently in the Qdrant collection"""
    print("\nChecking current Qdrant collection content...")

    from backend.vector_store.retriever import QdrantRetriever

    try:
        retriever = QdrantRetriever()

        # Use Qdrant's count method to get total number of documents
        from qdrant_client.http import models
        count_response = retriever.client.count(
            collection_name=retriever.collection_name
        )

        print(f"Current document count: {count_response.count}")
        return count_response.count

    except Exception as e:
        print(f"[ERROR] Could not check Qdrant collection: {e}")
        return 0

def main():
    print("Textbook Content Ingestion Tool")
    print("="*50)

    # First, check current content
    current_count = check_current_content()
    print(f"Current document count in Qdrant: {current_count}")

    # Run ingestion to ensure all content is there
    success = run_sitemap_ingestion()

    if success:
        new_count = check_current_content()
        print(f"New document count in Qdrant: {new_count}")

        if new_count > current_count:
            print(f"\n[SUCCESS] Added {new_count - current_count} new documents to Qdrant!")
        else:
            print(f"\n[INFO] Document count unchanged. Content may already be fully ingested.")

        print("\nThe RAG system now has the complete textbook content in the vector database.")
        print("The chatbot should provide accurate, textbook-based responses.")
    else:
        print("\n[ERROR] Ingestion failed. Please check your environment variables and network connection.")

if __name__ == "__main__":
    main()