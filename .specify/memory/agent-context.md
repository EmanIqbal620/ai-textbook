# Humanoid Robotics AI Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-12-19

## Active Technologies

- FastAPI: Web framework for building the API
- OpenAI Agents SDK: For RAG agent functionality and response generation
- Cohere API: For text embeddings generation
- Qdrant Cloud: Vector database for document retrieval
- Neon Serverless Postgres: For logging and analytics
- Python 3.11+: Runtime environment
- Docker: Containerization for deployment
- AsyncPG: Async PostgreSQL client

## Project Structure

```text
backend/
├── server.py: Main FastAPI application
├── api/
│   ├── __init__.py
│   ├── chat.py: Chat API endpoints
├── agents/
│   ├── __init__.py
│   ├── rag_agent.py: OpenAI Agents SDK integration
├── vector_store/
│   ├── __init__.py
│   ├── retriever.py: Qdrant integration and retrieval logic
├── database/
│   ├── __init__.py
│   ├── postgres_client.py: Neon Postgres integration
├── utils/
│   ├── __init__.py
│   ├── embeddings.py: Cohere embedding utilities
│   └── logging.py: Structured logging
├── requirements.txt: Python dependencies
├── DEPLOYMENT.md: Deployment configuration
└── .env.example: Environment variable template
specs/
└── 1-rag-agent-openai/
    ├── spec.md: Feature specification
    ├── plan.md: Implementation plan
    ├── research.md: Research and decisions
    ├── data-model.md: Data models
    ├── quickstart.md: Quickstart guide
    └── contracts/
        └── openapi.yaml: API contracts
```

## Commands

# Start the FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Install dependencies
pip install -r requirements.txt

# Test the chat endpoint
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is humanoid robotics?"}'

# Check health
curl http://localhost:8000/api/v1/health

## Code Style

# Python code follows PEP 8 guidelines
# Use async/await for all I/O operations
# Implement proper error handling with logging
# Use Pydantic for request/response validation
# Follow service layer pattern for business logic
# Use environment variables for configuration

## Recent Changes

- RAG Agent with OpenAI Agents SDK: Added FastAPI endpoints, OpenAI Agents integration, Qdrant retrieval, Neon Postgres logging
- Vector Store Integration: Implemented Qdrant client for document retrieval
- Database Layer: Added Neon Postgres client for logging and analytics

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->