# Data Model: Website Ingestion and Vector Storage

## Core Entities

### TextChunk
A segment of extracted content from a documentation page, with associated metadata

**Fields:**
- `id` (string): Unique identifier for the chunk
- `content` (string): The extracted text content (500-1000 tokens)
- `url` (string): The source URL of the original page
- `title` (string): The page title from the HTML
- `section` (string): The section identifier from the textbook structure
- `module` (string): The module identifier from the textbook structure
- `chunk_index` (integer): The sequential position of this chunk in the original content
- `created_at` (datetime): Timestamp when the chunk was created
- `embedding` (list[float]): The vector representation of the content

**Validation Rules:**
- Content must be between 500-1000 tokens
- URL must be a valid page from the textbook site
- Chunk index must be non-negative
- Embedding must have the correct dimension for the Cohere model used
- Section and module must be valid textbook structure identifiers

### DeploymentValidation
Represents the validation of deployed website availability

**Fields:**
- `base_url` (string): The base URL of the deployed website
- `status` (string): Success or failure status
- `accessible_urls` (list[string]): List of accessible URLs discovered
- `validated_at` (datetime): Timestamp when validation occurred
- `error_message` (string, optional): Error details if validation failed

**Validation Rules:**
- Status must be either "success" or "failure"
- If status is "success", accessible_urls must be present and non-empty
- If status is "failure", error_message must be present
- Base URL must be a valid URL

### CrawlResult
Represents the outcome of crawling a single page

**Fields:**
- `url` (string): The URL that was crawled
- `status` (string): Success or failure status
- `title` (string): Extracted page title
- `content_length` (integer): Length of extracted content in characters
- `module` (string): Module identifier from the URL structure
- `section` (string): Section identifier from the URL structure
- `crawled_at` (datetime): Timestamp when crawling occurred
- `error_message` (string, optional): Error details if crawling failed

**Validation Rules:**
- Status must be either "success" or "failure"
- If status is "failure", error_message must be present
- URL must be a valid URL
- Module and section must be properly extracted from the URL structure

### TextExtractionResult
Represents the outcome of extracting text from a web page

**Fields:**
- `url` (string): The source URL of the content
- `original_content_length` (integer): Length of original HTML content
- `extracted_content` (string): The extracted plain text content
- `extracted_content_length` (integer): Length of extracted content
- `extraction_steps_applied` (list[string]): List of extraction operations applied
- `extracted_at` (datetime): Timestamp when extraction occurred
- `status` (string): Success or failure status
- `error_message` (string, optional): Error details if extraction failed

**Validation Rules:**
- Status must be either "success" or "failure"
- If status is "success", extracted_content must be present
- If status is "failure", error_message must be present
- Extracted content should be less than original content length (due to HTML removal)

### TextCleanResult
Represents the outcome of cleaning text content

**Fields:**
- `original_url` (string): The source URL of the content
- `original_content_length` (integer): Length of original content
- `cleaned_content` (string): The cleaned text content
- `cleaned_content_length` (integer): Length of cleaned content
- `cleaning_steps_applied` (list[string]): List of cleaning operations applied
- `cleaned_at` (datetime): Timestamp when cleaning occurred
- `status` (string): Success or failure status
- `error_message` (string, optional): Error details if cleaning failed

**Validation Rules:**
- Status must be either "success" or "failure"
- If status is "success", cleaned_content must be present
- If status is "failure", error_message must be present
- Cleaned content should be less than or equal to original content length

### EmbeddingResult
Represents the outcome of generating an embedding

**Fields:**
- `chunk_id` (string): Reference to the text chunk that was embedded
- `embedding_vector` (list[float]): The generated embedding vector
- `model_used` (string): The Cohere model used for embedding
- `embedding_created_at` (datetime): Timestamp when embedding was generated
- `status` (string): Success or failure status
- `error_message` (string, optional): Error details if embedding failed

**Validation Rules:**
- Status must be either "success" or "failure"
- If status is "success", embedding_vector must be present and properly sized
- If status is "failure", error_message must be present
- Model used must be a valid Cohere embedding model

### VectorStoreRecord
Represents a record stored in Qdrant

**Fields:**
- `id` (string): Unique identifier (matches TextChunk.id)
- `vector` (list[float]): The embedding vector to store
- `payload` (dict): Metadata including URL, title, section, module, chunk_index
- `stored_at` (datetime): Timestamp when stored in Qdrant

**Validation Rules:**
- ID must be unique within the collection
- Vector must match the expected dimension for the collection
- Payload must contain required metadata fields (URL, section, module)

### RetrievalValidation
Represents the validation of retrieval functionality

**Fields:**
- `test_query` (string): The query used for validation
- `retrieved_chunks` (list[dict]): List of retrieved chunk data
- `validation_passed` (boolean): Whether the validation succeeded
- `similarity_threshold` (float): Minimum similarity threshold used
- `validation_at` (datetime): Timestamp when validation occurred
- `notes` (string, optional): Additional validation notes

**Validation Rules:**
- Retrieved chunks must contain valid chunk data
- Validation passed must be true or false
- Similarity threshold must be between 0 and 1

### IngestionLog
Represents logging information stored in Neon database for ingestion operations

**Fields:**
- `id` (integer): Auto-incrementing primary key
- `operation_type` (string): Type of operation (crawl, extract, clean, embed, store, validate)
- `url` (string, optional): URL being processed (if applicable)
- `status` (string): Success or failure status
- `message` (text): Detailed log message
- `error_details` (text, optional): Error details if operation failed
- `timestamp` (datetime): When the operation was logged
- `duration_ms` (integer, optional): Duration of the operation in milliseconds

**Validation Rules:**
- Operation type must be one of the defined values
- Status must be either "success" or "failure"
- Timestamp must be current or past time

### RetrievalLog
Represents logging information stored in Neon database for retrieval operations

**Fields:**
- `id` (integer): Auto-incrementing primary key
- `query_text` (text): The search query used
- `retrieval_count` (integer): Number of items retrieved
- `similarity_threshold` (float): Minimum similarity used for retrieval
- `status` (string): Success or failure status
- `message` (text): Detailed log message
- `error_details` (text, optional): Error details if operation failed
- `timestamp` (datetime): When the operation was logged
- `duration_ms` (integer, optional): Duration of the operation in milliseconds

**Validation Rules:**
- Status must be either "success" or "failure"
- Similarity threshold must be between 0 and 1
- Timestamp must be current or past time