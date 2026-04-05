import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="backend/.env")

# Add backend to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def check_qdrant_content():
    """Check if Qdrant has textbook content"""
    from backend.vector_store.retriever import QdrantRetriever

    print("Checking Qdrant collection for textbook content...")

    try:
        # Initialize the retriever
        retriever = QdrantRetriever()

        # Check if collection exists and is healthy
        if not retriever.is_healthy():
            print("[ERROR] Qdrant collection is not healthy or accessible")
            return False

        print("[OK] Qdrant connection is healthy")

        # Try a simple search to see if there's content
        test_results = await retriever.search("humanoid robotics", top_k=5)

        if len(test_results) == 0:
            print("[INFO] No content found in Qdrant collection - the textbook content may not have been ingested yet")
            print("This explains why the chatbot returns 'This topic is not covered in the book yet'")
            return False
        else:
            print(f"[OK] Found {len(test_results)} documents in Qdrant collection")
            print("Sample content from first document:")
            if test_results:
                first_doc = test_results[0]
                print(f"Content preview: {first_doc['content'][:200]}...")
                print(f"Metadata: {first_doc['metadata']}")
            return True

    except Exception as e:
        print(f"[ERROR] Failed to check Qdrant content: {e}")
        return False

async def check_ingestion_status():
    """Check if the ingestion process has been run"""
    print("\nChecking ingestion status...")

    # Check if the sitemap ingestion has been run
    sitemap_url = os.getenv("BOOK_BASE_URL", "https://humanoid-robotics-textbook-4ufa.vercel.app/")
    print(f"Expected sitemap URL: {sitemap_url}/sitemap.xml")

    # Try to fetch the sitemap to see if it's accessible
    import requests
    try:
        response = requests.get(f"{sitemap_url}/sitemap.xml", timeout=10)
        if response.status_code == 200:
            print("[OK] Sitemap is accessible")
            # Count URLs in sitemap
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
            urls = root.findall(f"{namespace}url")
            print(f"Found {len(urls)} URLs in sitemap")
        else:
            print(f"[ERROR] Sitemap not accessible, status code: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Could not access sitemap: {e}")

async def main():
    print("Checking RAG System Status...")
    print("="*60)

    # Check Qdrant content
    content_ok = await check_qdrant_content()

    # Check ingestion status
    await check_ingestion_status()

    print("\n" + "="*60)
    if content_ok:
        print("[SUCCESS] Qdrant has textbook content - RAG system should work properly")
    else:
        print("[ISSUE IDENTIFIED] Qdrant collection is empty - need to run ingestion")
        print("\nTo fix this issue, you need to:")
        print("1. Run the sitemap ingestion script to populate Qdrant with textbook content")
        print("2. The script is located at: backend/ingestion/sitemap_ingestion.py")
        print("3. Make sure your sitemap URL is correct in the ingestion script")
        print("\nCommand to run ingestion:")
        print("cd backend && python -c \"from ingestion.sitemap_ingestion import main; main()\"")

if __name__ == "__main__":
    asyncio.run(main())