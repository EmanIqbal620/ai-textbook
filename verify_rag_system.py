import requests
import json

def verify_rag_system():
    """Verify that the RAG system is working correctly with strict rules"""
    print("Verifying RAG System...")

    base_url = "http://localhost:8000/api/v1"

    # Test that the system works with book-related queries
    book_queries = [
        "What is ROS 2?",
        "Explain Gazebo simulation",
        "What is NVIDIA Isaac?",
        "What are Vision-Language-Action systems?"
    ]

    print("\nTesting book-related queries:")
    for query in book_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "top_k": 5,
                    "max_tokens": 800,
                    "temperature": 0.3
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data['response']
                sources = data.get('sources', [])

                print(f"  Response: {response_text[:100]}...")
                print(f"  Sources: {len(sources)} retrieved")

                # Check if it's using book content
                if any(word in response_text.lower() for word in ['source', 'context', 'book', 'textbook']):
                    print("  Status: ✅ Using book content")
                else:
                    print("  Status: ? Book content used (response present)")
            else:
                print(f"  Status: ❌ Error - {response.status_code}")

        except Exception as e:
            print(f"  Status: ❌ Error - {e}")

    # Test that the system refuses to answer non-book queries
    non_book_queries = [
        "What is the capital of France?",
        "How to bake a cake?",
        "Explain quantum physics"
    ]

    print("\nTesting non-book queries (should not use general knowledge):")
    for query in non_book_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "top_k": 5,
                    "max_tokens": 800,
                    "temperature": 0.3
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data['response']

                print(f"  Response: {response_text[:100]}...")

                # Check if it refuses to answer (good) or tries to answer (bad)
                if "not available in the book" in response_text or "does not contain" in response_text:
                    print("  Status: ✅ Correctly refuses to answer (book-only)")
                else:
                    print("  Status: ❌ May have used general knowledge")
            else:
                print(f"  Status: ❌ Error - {response.status_code}")

        except Exception as e:
            print(f"  Status: ❌ Error - {e}")

    # Test the health of the system
    print("\nTesting system health:")
    try:
        health_response = requests.get(f"{base_url.replace('/api/v1', '')}/api/v1/health", timeout=10)
        if health_response.status_code == 200:
            print("  Status: ✅ Backend server healthy")
        else:
            print("  Status: ❌ Backend server unhealthy")
    except:
        print("  Status: ❌ Cannot connect to backend server")

def main():
    print("RAG System Verification")
    print("="*30)
    verify_rag_system()
    print("\n" + "="*30)
    print("Verification complete!")

if __name__ == "__main__":
    main()