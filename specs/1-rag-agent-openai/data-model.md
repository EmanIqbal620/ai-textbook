# Data Model: RAG Agent with OpenAI Agents SDK and FastAPI

## Entity: ChatInteraction

### Fields
- `id` (string, auto-generated): Unique identifier for the interaction
- `query` (string, required): The original user query
- `response` (string, required): The generated response from the RAG agent
- `context` (string, required): The retrieved context used for response generation
- `sources` (array of objects, required): List of source documents used
- `response_time_ms` (integer, required): Time taken to process the request in milliseconds
- `user_id` (string, optional): Identifier for the user (if authenticated)
- `timestamp` (datetime, required): When the interaction occurred

### Validation Rules
- `query` must be between 1-2000 characters
- `response` must not be empty
- `sources` array must contain at least 1 source when context is found
- `response_time_ms` must be positive

### Relationships
- Each interaction is logged to the Neon Postgres database
- Sources reference documents in the Qdrant vector database

## Entity: SourceDocument

### Fields
- `id` (string, required): Unique identifier for the source document in Qdrant
- `source` (string, required): URL or file path of the source
- `score` (float, required): Relevance score from vector similarity search
- `content` (string, required): The content of the document chunk
- `metadata` (object, optional): Additional metadata about the document

### Validation Rules
- `score` must be between 0 and 1
- `content` must not be empty
- `id` must be a valid Qdrant document ID

### Relationships
- Referenced by ChatInteraction as part of the sources array
- Stored in Qdrant vector database with embedding vectors

## Entity: UserSession (Future Enhancement)

### Fields
- `session_id` (string, required): Unique identifier for the session
- `history` (array of ChatInteraction, optional): Conversation history
- `preferences` (object, optional): User preferences for the chat experience

### Validation Rules
- `session_id` must be unique
- `history` array has maximum size of 50 interactions

### Relationships
- Contains multiple ChatInteraction entities
- Optional entity for future conversation memory features

## Entity: QueryMetrics

### Fields
- `id` (string, auto-generated): Unique identifier for the metric record
- `query` (string, required): The original user query
- `response_time_ms` (integer, required): Time taken to process the request
- `success` (boolean, required): Whether the query was processed successfully
- `timestamp` (datetime, required): When the query was processed
- `error_message` (string, optional): Error message if query failed
- `user_id` (string, optional): Identifier for the user (if authenticated)

### Validation Rules
- `response_time_ms` must be positive
- `success` must be true or false
- `error_message` only present when success is false

### Relationships
- Created for every query attempt (successful or failed)
- Stored in Neon Postgres database for analytics