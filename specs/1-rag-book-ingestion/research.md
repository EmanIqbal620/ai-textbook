# Research: Website Ingestion and Vector Storage

## Decision: Project Management Tool
**Rationale**: uv was selected for project management because:
- Modern, fast Python package manager and resolver
- Better performance than pip + requirements.txt
- Built-in virtual environment management
- Growing adoption in the Python community

## Decision: Technology Stack
**Rationale**: Python 3.11 was selected as the primary language because:
- Rich ecosystem for web crawling, text processing, and ML/AI tasks
- Excellent libraries for HTML parsing (BeautifulSoup) and HTTP requests (requests)
- Strong support for embedding generation and vector databases
- Alignment with the AI/ML focus of the textbook

## Decision: Website Deployment Validation
**Rationale**: Using requests for deployment validation because:
- Standard library for HTTP operations in Python
- Can validate site availability and accessible URLs
- Lightweight and reliable for checking deployed resources

## Decision: Text Extraction Strategy
**Rationale**: Using BeautifulSoup4 for text extraction because:
- Can effectively strip HTML tags while preserving content structure
- Handles complex HTML structures from Docusaurus sites well
- Allows for selective content extraction (e.g., ignoring navigation, headers)
- Maintains readability of the extracted text

## Decision: Text Cleaning Strategy
**Rationale**: Using BeautifulSoup4 + custom cleaning functions because:
- Can effectively strip HTML tags while preserving content structure
- Handles complex HTML structures from Docusaurus sites well
- Allows for selective content extraction (e.g., ignoring navigation, headers)
- Additional cleaning functions can normalize text and remove artifacts

## Decision: Text Chunking Strategy
**Rationale**: Using a token-based chunking approach with overlap because:
- Ensures chunks are of appropriate size for embedding models (500-1000 tokens)
- Overlap (20%) maintains context across chunk boundaries
- Prevents information fragmentation during retrieval
- Standard approach in RAG systems

## Decision: Embedding Provider
**Rationale**: Cohere was selected as the embedding provider because:
- Specified in the feature requirements
- High-quality embeddings suitable for semantic search
- Good API reliability and rate limits
- Supports educational use cases

## Decision: Vector Database
**Rationale**: Qdrant Cloud Free Tier was selected because:
- Specified in the feature requirements
- Designed specifically for vector storage and similarity search
- Cloud-hosted option reduces infrastructure complexity
- Supports metadata storage for URL, section, module, etc.

## Decision: Logging Database
**Rationale**: Neon PostgreSQL was selected for logging because:
- Specified in the feature requirements (step 8 uses Neon for logging only)
- Reliable and scalable PostgreSQL-compatible database
- Good for structured logging of ingestion status and validation results
- Separates logging data from vector storage (Qdrant) for better architecture

## Decision: Retrieval Validation
**Rationale**: Using similarity queries for validation because:
- Tests actual retrieval functionality
- Validates embeddings are properly stored and accessible
- Confirms semantic search works as expected

## Decision: Logging Strategy
**Rationale**: Structured logging to Neon database for ingestion, embedding, and retrieval because:
- Provides persistent, queryable logs of all operations
- Enables monitoring of success/failure rates
- Helps with debugging and performance optimization
- Tracks important metrics for each operation with searchable history

## Alternatives Considered

### Project Management Alternatives:
- pip + requirements.txt: More traditional but slower than uv
- Poetry: Good but uv is newer and faster
- Conda: More complex for this use case

### Website Validation Alternatives:
- Selenium: Overkill for simple availability checks
- Custom deployment scripts: More complex than needed

### Text Extraction Alternatives:
- Newspaper3k: More focused on news articles than documentation
- Trafilatura: Good but BeautifulSoup is more familiar and sufficient
- LangChain Document Loaders: Possible but more complex than needed

### Text Cleaning Alternatives:
- Newspaper3k: More focused on news articles than documentation
- Trafilatura: Good but BeautifulSoup with custom logic is more flexible
- LangChain Document Loaders: Possible but more complex than needed

### Chunking Alternatives:
- Character-based: Less context-aware than token-based
- Sentence-based: May create uneven chunks
- Recursive: Standard but token-based is more precise

### Embedding Alternatives:
- OpenAI: Not specified in requirements
- Hugging Face: Self-hosted complexity not needed
- Google: Not specified in requirements

### Logging Database Alternatives:
- File-based logging: Less queryable and scalable
- SQLite: Local only, less suitable for production
- MongoDB: Different paradigm than needed for logging

### Retrieval Validation Alternatives:
- Simple count verification: Doesn't test actual retrieval
- Manual validation: Not scalable or automated