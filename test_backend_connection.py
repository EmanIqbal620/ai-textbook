import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_backend_connection():
    """Test if the backend RAG API is running and accessible"""
    print("Testing backend RAG API connection...")

    # Check if the RAG API URL is set
    rag_api_url = os.getenv('RAG_API_URL', 'http://localhost:8000/api')
    print(f"Testing RAG API at: {rag_api_url}")

    # Test health endpoint
    try:
        health_url = f"{rag_api_url.replace('/api', '')}/api/v1/health"
        print(f"Testing health endpoint: {health_url}")
        response = requests.get(health_url, timeout=10)

        if response.status_code == 200:
            health_data = response.json()
            print(f"[OK] Health check passed: {health_data}")
        else:
            print(f"[ERROR] Health check failed with status: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to backend at {rag_api_url}")
        print("The backend server may not be running.")
        print("To start the backend server, run:")
        print("cd backend && python server.py")
        return False
    except Exception as e:
        print(f"[ERROR] Health check failed with error: {e}")
        return False

    # Test chat endpoint
    try:
        chat_url = f"{rag_api_url}/v1/chat"
        print(f"Testing chat endpoint: {chat_url}")

        # Prepare test data
        test_data = {
            "query": "What is humanoid robotics?",
            "user_id": "test_user",
            "top_k": 3,
            "max_tokens": 500,
            "temperature": 0.7
        }

        response = requests.post(
            chat_url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            chat_data = response.json()
            print(f"[OK] Chat endpoint working")
            print(f"Response preview: {chat_data.get('response', '')[:100]}...")
        else:
            print(f"[ERROR] Chat endpoint failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"[ERROR] Chat endpoint test failed with error: {e}")
        return False

    return True

def test_frontend_config():
    """Test frontend configuration"""
    print("\nTesting frontend configuration...")

    # Check if the frontend is using the correct backend URL
    print("Current RAG_API_URL from environment:", os.getenv('RAG_API_URL', 'http://localhost:8000/api'))

    # Check the chatbot service file
    try:
        with open('humanoid-robotics-textbook/src/services/chatbotService.js', 'r') as f:
            content = f.read()
            if 'http://localhost:8000/api' in content:
                print("[INFO] Frontend is configured to use local backend")
            else:
                print("[INFO] Frontend may be using a different backend URL")
    except FileNotFoundError:
        print("[ERROR] Chatbot service file not found")

def main():
    print("Testing RAG Backend Connection...")
    print("="*60)

    backend_ok = test_backend_connection()
    test_frontend_config()

    print("\n" + "="*60)
    if backend_ok:
        print("[SUCCESS] Backend RAG API is working properly")
        print("The issue may be that the backend server is not running")
        print("\nTo start the backend server:")
        print("cd backend && python server.py")
        print("\nThen restart the frontend with:")
        print("cd humanoid-robotics-textbook && npm run start")
    else:
        print("[ISSUE] Backend RAG API is not accessible")
        print("Make sure the backend server is running on port 8000")

if __name__ == "__main__":
    main()