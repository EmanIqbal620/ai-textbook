# Quickstart: RAG Chatbot for Textbook

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (for vector database)
- An LLM API key (OpenAI, Anthropic, etc.)

### Backend Setup
1. Navigate to the backend directory
2. Install Python dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```
4. Start the backend: `uvicorn main:app --reload`

### Frontend Setup
1. Navigate to the frontend directory
2. Install dependencies: `npm install`
3. Start the development server: `npm start`

## Running the RAG Chatbot

### 1. Ingest Textbook Content
Before the chatbot can answer questions, you need to ingest the textbook content:
```bash
python -m scripts.ingest_textbook --source path/to/textbook
```

### 2. Start a Conversation
1. Launch the frontend application
2. The app will automatically create a new session
3. Type your question in the chat interface
4. The chatbot will retrieve relevant content and generate a response

### 3. API Usage Example
```bash
# Start a new session
curl -X POST http://localhost:8000/api/chat/start

# Ask a question (replace {sessionId} with actual session ID)
curl -X POST http://localhost:8000/api/chat/{sessionId}/question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a ROS node?"}'
```

## Testing the System
- Run backend tests: `pytest tests/`
- Run frontend tests: `npm test`
- Integration tests: `pytest tests/integration/`

## Validation
To validate the system works correctly:
1. Ask a question about content in the textbook - should get a relevant answer
2. Ask a question about content not in the textbook - should respond with "This topic is not covered in the book yet."
3. Verify responses are in simple English and beginner-friendly
4. Check that no external knowledge is used in responses