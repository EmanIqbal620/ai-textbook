# Quickstart Guide: RAG Agent with OpenAI Agents SDK and FastAPI

## Prerequisites

- Python 3.11+
- OpenAI API key
- Cohere API key
- Qdrant Cloud URL and API key
- Neon Postgres database URL
- Git

## Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-name>/backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
NEON_DATABASE_URL=your_neon_database_url_here
QDRANT_COLLECTION_NAME=humanoid_ai_book
```

## Running Locally

### 1. Start the FastAPI Server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test the API
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Chat endpoint
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is humanoid robotics?",
    "top_k": 5,
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

## API Endpoints

### Chat Endpoint
`POST /api/v1/chat`

Request body:
```json
{
  "query": "Your question here",
  "user_id": "optional_user_id",
  "max_tokens": 1000,
  "temperature": 0.7,
  "top_k": 5,
  "user_selected_text": "optional selected text"
}
```

Response:
```json
{
  "response": "Generated response",
  "sources": [
    {
      "id": "source_id",
      "source": "source_url",
      "score": 0.85,
      "page_content": "Truncated content..."
    }
  ],
  "response_time": 1.25,
  "query": "Your question here"
}
```

### Health Check
`GET /api/v1/health`

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-19T10:00:00Z"
}
```

### Usage Statistics
`GET /api/v1/stats`

Response:
```json
{
  "total_interactions": 1500,
  "total_errors": 12,
  "average_response_time_ms": 1250.5,
  "recent_interactions": 45,
  "timestamp": "2025-12-19T10:00:00Z"
}
```

## Environment Configuration

### Development
- Use `--reload` flag with uvicorn for auto-reload on code changes
- Enable detailed logging for debugging

### Production
- Set appropriate environment variables for production
- Configure reverse proxy (nginx, etc.) for SSL termination
- Set up proper logging aggregation
- Configure monitoring and alerting

## Troubleshooting

### Common Issues

1. **API Keys Not Working**
   - Verify all API keys are correctly set in environment variables
   - Check for typos in the .env file

2. **Qdrant Connection Issues**
   - Verify QDRANT_URL and QDRANT_API_KEY are correct
   - Ensure the collection name matches what was used during ingestion

3. **Database Connection Issues**
   - Verify NEON_DATABASE_URL is properly formatted
   - Check that the database is accessible from your environment

4. **Slow Response Times**
   - Check vector database query performance
   - Verify that embeddings are properly indexed
   - Monitor API rate limits for OpenAI and Cohere