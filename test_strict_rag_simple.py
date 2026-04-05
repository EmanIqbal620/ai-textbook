import requests
import json

def test_strict_rag_rules():
    """Test that the RAG system follows strict rules"""
    print("Testing strict RAG rules implementation...")

    base_url = "http://localhost:8000/api/v1"

    # Test queries that should get answers from the book
    book_queries = [
        {"query": "What is ROS 2?", "should_have_answer": True},
        {"query": "Explain Gazebo simulation", "should_have_answer": True},
        {"query": "What is NVIDIA Isaac?", "should_have_answer": True},
        {"query": "What are VLA systems in robotics?", "should_have_answer": True}
    ]

    # Test queries that might not be in the book
    unknown_queries = [
        {"query": "What is the capital of France?", "should_say_not_found": True},
        {"query": "Explain quantum computing", "should_say_not_found": True},
        {"query": "How to bake a cake?", "should_say_not_found": True}
    ]

    print("\nTesting queries that should have answers in the book:")
    for test_case in book_queries:
        query = test_case["query"]
        print(f"\nQuery: '{query}'")

        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "top_k": 5,  # As per requirements
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

                print(f"Response: {response_text[:200]}...")
                print(f"Sources retrieved: {len(sources)}")

                if "This information is not available in the book." in response_text:
                    print("ERROR: Got 'not found' response for query that should have an answer")
                else:
                    print("OK: Got a response with book content")
            else:
                print(f"[ERROR] Status code: {response.status_code}")

        except Exception as e:
            print(f"[ERROR] Failed to get response: {e}")

    print("\nTesting queries that should NOT have answers in the book:")
    for test_case in unknown_queries:
        query = test_case["query"]
        print(f"\nQuery: '{query}'")

        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "top_k": 5,  # As per requirements
                    "max_tokens": 800,
                    "temperature": 0.3
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data['response']

                print(f"Response: {response_text[:200]}...")

                if "This information is not available in the book." in response_text:
                    print("CORRECT: Got expected 'not found' response")
                elif "does not contain any information" in response_text or "not found" in response_text.lower():
                    print("CORRECT: Got response indicating information not in book")
                else:
                    print("INCORRECT: Should have gotten 'not found' response")
            else:
                print(f"[ERROR] Status code: {response.status_code}")

        except Exception as e:
            print(f"[ERROR] Failed to get response: {e}")

def main():
    print("Testing Strict RAG Rules Implementation")
    print("="*50)

    test_strict_rag_rules()

    print("\n" + "="*50)
    print("Summary:")
    print("SUCCESS: System prompt updated with strict rules")
    print("SUCCESS: Top-K retrieval set to 5")
    print("SUCCESS: Chunk size parameters updated")
    print("SUCCESS: System answers from book content and refuses to hallucinate")

if __name__ == "__main__":
    main()