# Spec 3 – RAG Agent with OpenAI Agents SDK and FastAPI

## Goal
Build a production-ready Retrieval-Augmented Generation (RAG) chatbot using the OpenAI Agents SDK and FastAPI that can answer questions about the published Docusaurus book.

## Scope
- Use Qdrant Cloud as the vector database for retrieval
- Use Cohere embeddings already generated in previous specs
- Use Neon Serverless Postgres for logging queries and responses
- Expose a FastAPI HTTP endpoint for chatbot queries
- Ensure the chatbot works both locally and after deployment
- Support answering questions based on:
  1) Full book content
  2) User-selected text from the Docusaurus website

## Constraints
- Follow Spec-Driven Development strictly
- No hard-coded secrets; use environment variables
- The deployed chatbot must work with the deployed Docusaurus site
- The system must gracefully handle missing context or retrieval failures

## Success Criteria
- FastAPI endpoint responds correctly after deployment
- RAG retrieval pulls relevant context from the book
- Responses are properly logged in Neon Postgres
- System handles errors gracefully
- Response times are under 5 seconds for 95% of requests

## Architecture

### Components
1. **FastAPI Application** - Main web server and API endpoint
2. **OpenAI Agents SDK** - Agent framework for RAG logic
3. **Qdrant Cloud** - Vector database for document embeddings
4. **Neon Serverless Postgres** - Logging and metadata storage
5. **Cohere API** - Embedding generation for queries
6. **Environment Configuration** - Secure secret management

### Data Flow
1. User sends query to FastAPI endpoint
2. Query is embedded using Cohere API
3. Vector search performed in Qdrant Cloud
4. Relevant context retrieved from vector store
5. OpenAI Agent processes context and generates response
6. Query, context, and response logged in Neon Postgres
7. Response returned to user

## Implementation Details

### Environment Variables
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_database_url
QDRANT_COLLECTION_NAME=humanoid_ai_book
```

### FastAPI Endpoints
```
POST /chat - Main chat endpoint
GET /health - Health check endpoint
POST /query - Alternative query endpoint with more options
```

### Database Schema (Neon Postgres)
```sql
CREATE TABLE chat_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    context TEXT,
    sources JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    user_id VARCHAR(255)
);

CREATE TABLE query_metrics (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    response_time_ms INTEGER,
    success BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT
);
```

### Error Handling
- Graceful degradation when vector store is unavailable
- Fallback responses when no relevant context found
- Proper logging of all errors for debugging
- Rate limiting to prevent abuse

## Security Considerations
- All API keys stored in environment variables
- Input validation to prevent injection attacks
- Rate limiting to prevent abuse
- Query sanitization before database logging

## Deployment Configuration
- Docker container for consistent deployment
- Environment-based configuration
- Health checks for monitoring
- Logging configuration for production

## Testing Strategy
- Unit tests for individual components
- Integration tests for full RAG pipeline
- Performance tests for response times
- End-to-end tests for deployment validation

## Monitoring and Observability
- Response time metrics
- Error rate tracking
- Query volume monitoring
- Vector store health checks

## Dependencies
- fastapi
- uvicorn
- openai
- cohere
- qdrant-client
- asyncpg (for Neon Postgres)
- pydantic
- python-dotenv
- pytest
- httpx (for testing)

## Performance Requirements
- 95% of queries should respond in under 5 seconds
- Support for 100 concurrent users
- Vector search should complete in under 1 second
- Efficient memory usage to handle long-running processes