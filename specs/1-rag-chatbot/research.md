# Research: RAG Chatbot for Textbook

## Decision: LLM Provider Selection
**Rationale**: For the RAG chatbot, we'll use OpenAI GPT models as they provide good balance of quality, safety features, and ease of integration. Alternative options include Anthropic Claude, local models like Llama, or Azure OpenAI. OpenAI was selected for its strong safety features and reliable API.

## Decision: Vector Database
**Rationale**: Qdrant was selected as the vector database based on the existing project architecture (as mentioned in the README). It provides efficient similarity search capabilities needed for the RAG system and integrates well with Python-based backend services.

## Decision: Conversation Context Management
**Rationale**: For maintaining conversation context within sessions, we'll implement a session-based approach using in-memory storage or Redis. This allows context to persist during a single user session but clears when the session ends, respecting privacy requirements while enabling natural conversations.

## Decision: Frontend Framework
**Rationale**: React was selected as the frontend framework based on industry standards and its component-based architecture which is well-suited for building interactive chat interfaces. It integrates well with the Docusaurus-based textbook platform.

## Decision: Backend Framework
**Rationale**: FastAPI was selected for the backend due to its performance, automatic API documentation generation, and strong typing support. It's well-suited for building API services that handle RAG operations.

## Decision: Content Retrieval Strategy
**Rationale**: The system will use semantic search to retrieve relevant textbook sections. This involves embedding textbook content during an ingestion process and using cosine similarity to find the most relevant chunks when a question is asked. This approach provides better results than keyword-based search for complex questions.

## Decision: Content Validation and Filtering
**Rationale**: To ensure compliance with constitution rules (no external knowledge, no hallucination), the system will implement strict content validation that verifies all responses are grounded in the retrieved context. If no relevant context is found, the system will respond with "This topic is not covered in the book yet."

## Alternatives Considered
- For LLM: Local models (higher maintenance, less reliable), Azure OpenAI (vendor lock-in), Anthropic (good alternative but OpenAI has broader ecosystem)
- For vector DB: Pinecone (managed but more expensive), Weaviate (good alternative but Qdrant already mentioned in project)
- For frontend: Vue, Angular (React has broader ecosystem and component reusability)
- For backend: Flask (less performant, no automatic docs), Django (overkill for API service)