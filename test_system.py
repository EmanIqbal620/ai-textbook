import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        from backend.api.models import ChatRequest, ChatResponse, HealthResponse
        print("[OK] Models imported successfully")

        from backend.agents.rag_agent import RAGAgent
        print("[OK] RAG Agent module imported successfully")

        from backend.vector_store.retriever import QdrantRetriever
        print("[OK] Qdrant Retriever module imported successfully")

        from backend.database.postgres_client import PostgresService
        print("[OK] Postgres Service module imported successfully")

        from backend.utils.embeddings import EmbeddingService
        print("[OK] Embedding Service module imported successfully")

        print("\nAll modules imported successfully! The system structure is correct.")
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    try:
        # Test model creation
        from backend.api.models import ChatRequest, ChatResponse, HealthResponse
        from datetime import datetime

        # Create a sample request
        sample_request = ChatRequest(
            query="What is humanoid robotics?",
            user_id="test_user",
            max_tokens=500,
            temperature=0.7,
            top_k=3
        )
        print(f"[OK] ChatRequest model works: query='{sample_request.query}'")

        # Create a sample response
        sample_response = ChatResponse(
            response="Humanoid robotics is a field that focuses on creating robots with human-like characteristics.",
            sources=[{"id": "1", "source": "textbook", "score": 0.9}],
            response_time=1.25,
            query="What is humanoid robotics?"
        )
        print(f"[OK] ChatResponse model works: response='{sample_response.response[:50]}...'")

        # Create a health response
        health_response = HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow()
        )
        print(f"[OK] HealthResponse model works: status='{health_response.status}'")

        print("\n[OK] All Pydantic models are working correctly!")
        return True
    except Exception as e:
        print(f"[ERROR] Model test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG Agent System...")
    print("="*50)

    print("\n1. Testing imports...")
    imports_ok = test_imports()

    print("\n2. Testing basic functionality...")
    basic_ok = test_basic_functionality()

    print("\n" + "="*50)
    if imports_ok and basic_ok:
        print("[OK] All tests passed! The system is structurally sound.")
        print("\nNote: To fully test the RAG functionality, you need to:")
        print("- Add OPENAI_API_KEY to your .env file")
        print("- Add NEON_DATABASE_URL to your .env file")
        print("- Ensure the Qdrant vector database has been populated with textbook content")
        print("- Run the server with: uvicorn server:app --host 0.0.0.0 --port 8000")
    else:
        print("[ERROR] Some tests failed!")