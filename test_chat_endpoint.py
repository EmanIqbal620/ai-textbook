import requests
import json

def test_chat_endpoint():
    """Test the chat endpoint with a sample query"""
    print("Testing chat endpoint with sample queries...")

    base_url = "http://localhost:8000/api/v1"

    # Test queries related to robotics topics
    test_queries = [
        "What is humanoid robotics?",
        "Explain ROS 2",
        "Tell me about Gazebo simulation",
        "What is VLA in robotics?",
        "How does NVIDIA Isaac work?"
    ]

    for query in test_queries:
        print(f"\nTesting query: '{query}'")

        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "query": query,
                    "user_id": "test_user",
                    "top_k": 3,
                    "max_tokens": 500,
                    "temperature": 0.7
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                print(f"[SUCCESS] Response received")
                print(f"Response preview: {data['response'][:200]}...")
                if data['sources']:
                    print(f"Sources found: {len(data['sources'])} documents retrieved")
                else:
                    print("No sources found - might be using fallback response")
            else:
                print(f"[ERROR] Status code: {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"[ERROR] Failed to get response: {e}")

def main():
    print("Testing RAG Chat Endpoint...")
    print("="*60)

    test_chat_endpoint()

    print("\n" + "="*60)
    print("If the chat endpoint is working properly, the frontend chatbot")
    print("should now retrieve actual textbook content instead of fallback responses.")

if __name__ == "__main__":
    main()