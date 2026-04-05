import requests
import json

def test_content_accuracy():
    """Test if the content being retrieved is accurate and relevant"""
    print("Testing content accuracy and relevance...")

    base_url = "http://localhost:8000/api/v1"

    # Test specific robotics queries and check if the responses are from textbook content
    test_queries = [
        {"query": "What is ROS 2?", "expected_topic": "Robot Operating System"},
        {"query": "Explain Gazebo simulation", "expected_topic": "Gazebo simulator"},
        {"query": "What is NVIDIA Isaac?", "expected_topic": "NVIDIA Isaac platform"},
        {"query": "Vision Language Action robotics", "expected_topic": "VLA systems"},
        {"query": "humanoid robot definition", "expected_topic": "humanoid robotics"}
    ]

    for test_case in test_queries:
        query = test_case["query"]
        expected_topic = test_case["expected_topic"]

        print(f"\nTesting query: '{query}' (Expected topic: {expected_topic})")

        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "top_k": 5,
                    "max_tokens": 800,
                    "temperature": 0.3  # Lower temperature for more focused responses
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data['response']
                sources = data.get('sources', [])

                print(f"[SUCCESS] Response received")
                print(f"Response: {response_text[:300]}...")

                if sources:
                    print(f"Sources retrieved: {len(sources)}")
                    for i, source in enumerate(sources[:2]):  # Show first 2 sources
                        print(f"  Source {i+1}: Score: {source.get('score', 'N/A')}")
                        print(f"    Content preview: {source.get('page_content', '')[:100]}...")
                else:
                    print("No sources found")

                # Check if response contains expected topic
                response_lower = response_text.lower()
                expected_lower = expected_topic.lower()
                if expected_lower in response_lower:
                    print(f"[GOOD] Response contains expected topic: {expected_topic}")
                else:
                    print(f"[CONCERN] Response may not contain expected topic: {expected_topic}")

            else:
                print(f"[ERROR] Status code: {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"[ERROR] Failed to get response: {e}")

def test_qdrant_direct():
    """Test Qdrant directly to see what content is actually stored"""
    print("\n" + "="*60)
    print("Testing Qdrant content directly...")

    # Let's try a direct search to see what's in the database
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

    try:
        from backend.vector_store.retriever import QdrantRetriever
        import asyncio

        async def check_content():
            retriever = QdrantRetriever()

            # Search for specific robotics concepts
            search_terms = ["ROS 2", "Gazebo", "NVIDIA Isaac", "VLA", "humanoid robotics"]

            for term in search_terms:
                print(f"\nSearching for: '{term}'")
                results = await retriever.search(term, top_k=3)

                if results:
                    print(f"  Found {len(results)} results:")
                    for i, result in enumerate(results[:2]):  # Show first 2 results
                        print(f"    Result {i+1}:")
                        print(f"      Score: {result['score']}")
                        print(f"      Content preview: {result['content'][:150]}...")
                        print(f"      Metadata: {result['metadata']}")
                else:
                    print(f"  No results found for '{term}'")

        asyncio.run(check_content())

    except Exception as e:
        print(f"[ERROR] Could not access Qdrant directly: {e}")

def main():
    print("Testing Content Accuracy and Retrieval...")
    print("="*60)

    test_content_accuracy()
    test_qdrant_direct()

    print("\n" + "="*60)
    print("This test will help identify if:")
    print("1. The correct textbook content is in the database")
    print("2. The retrieval is working properly")
    print("3. The responses are based on actual textbook content")

if __name__ == "__main__":
    main()