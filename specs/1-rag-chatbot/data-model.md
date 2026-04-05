# Data Model: RAG Chatbot for Textbook

## Entities

### UserQuestion
- **id**: string (UUID)
- **content**: string (the actual question text)
- **timestamp**: datetime (when the question was asked)
- **sessionId**: string (reference to the conversation session)
- **processed**: boolean (whether the question has been processed)

### RetrievedContext
- **id**: string (UUID)
- **content**: string (the retrieved text chunk from textbook)
- **source**: string (reference to the textbook section/chapter)
- **similarityScore**: float (relevance score from vector search)
- **questionId**: string (reference to the original question)

### ChatbotResponse
- **id**: string (UUID)
- **content**: string (the generated response)
- **questionId**: string (reference to the original question)
- **timestamp**: datetime (when response was generated)
- **confidence**: enum (HIGH, MEDIUM, LOW, NONE)
- **sources**: array of strings (references to textbook sections used)

### ConversationSession
- **id**: string (UUID)
- **userId**: string (optional, for logged-in users)
- **startTime**: datetime
- **endTime**: datetime (null if active)
- **isActive**: boolean
- **messages**: array of Message objects

### Message
- **id**: string (UUID)
- **sessionId**: string (reference to conversation session)
- **role**: enum (USER, ASSISTANT)
- **content**: string
- **timestamp**: datetime
- **questionId**: string (for user messages)
- **responseId**: string (for assistant messages)

## Relationships
- ConversationSession has many Messages
- UserQuestion connects to RetrievedContext (via context retrieval)
- UserQuestion connects to ChatbotResponse (via question->response generation)
- RetrievedContext connects to ChatbotResponse (context used to generate response)

## Validation Rules
- UserQuestion.content must be non-empty
- ChatbotResponse.content must be grounded in RetrievedContext
- ConversationSession must have valid start time
- Message.role must be either USER or ASSISTANT

## State Transitions
- ConversationSession: ACTIVE → INACTIVE (when session ends)
- ChatbotResponse: PENDING → GENERATED → DELIVERED