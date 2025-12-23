# Quickstart: Website Ingestion and Vector Storage

## Prerequisites

- Python 3.11 or higher
- uv package manager
- Git (for cloning the repository)
- Cohere API key
- Qdrant Cloud account and API key
- Neon PostgreSQL database account and credentials

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Initialize Project with uv
```bash
# Install uv if not already installed
pip install uv

# Initialize the project (this creates pyproject.toml and uv.lock)
cd backend
uv init
```

### 3. Install Dependencies
```bash
# From the backend directory
uv pip install requests beautifulsoup4 cohere qdrant-client python-dotenv pytest psycopg2-binary
```

### 4. Configure Environment Variables
Create a `.env` file in the backend directory with:
```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DB_URL=your_neon_database_connection_string
TEXTBOOK_BASE_URL=https://humanoid-robotics-textbook-4ufa.vercel.app/
```

## Running the Ingestion Pipeline

### 1. Execute the Full Pipeline
```bash
cd backend
python -m src.main
```

### 2. Run Individual Components (Optional)
```bash
cd backend

# Validate deployed website
python -c "from src.rag_ingestion.deployment import validate_deployment; validate_deployment()"

# Crawl the textbook site
python -c "from src.rag_ingestion.crawler import crawl_textbook; crawl_textbook()"

# Extract book data from URLs
python -c "from src.rag_ingestion.text_extractor import extract_book_data; extract_book_data()"

# Clean and process text
python -c "from src.rag_ingestion.text_cleaner import clean_and_process_text; clean_and_process_text()"

# Process and chunk text
python -c "from src.rag_ingestion.chunker import process_chunks; process_chunks()"

# Generate embeddings
python -c "from src.rag_ingestion.embedder import generate_embeddings; generate_embeddings()"

# Store in Qdrant
python -c "from src.rag_ingestion.vector_store import store_vectors; store_vectors()"

# Validate retrieval
python -c "from src.rag_ingestion.vector_store import validate_retrieval; validate_retrieval()"
```

## Configuration Options

The pipeline can be configured through environment variables:

- `TEXTBOOK_BASE_URL`: Base URL of the textbook site (default: https://humanoid-robotics-textbook-4ufa.vercel.app/)
- `CHUNK_SIZE`: Maximum tokens per chunk (default: 750)
- `CHUNK_OVERLAP`: Overlap percentage between chunks (default: 0.2 for 20%)
- `COHERE_MODEL`: Cohere embedding model to use (default: embed-english-v3.0)
- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection (default: textbook_chunks)
- `SIMILARITY_THRESHOLD`: Minimum similarity for retrieval validation (default: 0.7)
- `NEON_DB_URL`: Connection string for Neon PostgreSQL database (for logging only)

## Verification

After running the pipeline, verify the results:

1. Check the number of vectors stored in Qdrant
2. Perform a sample retrieval to ensure vectors are stored correctly
3. Validate that metadata (URL, section, module) is preserved
4. Run retrieval validation using test similarity queries
5. Review logs in Neon database for ingestion, embedding, and retrieval results
6. Confirm that only logs are stored in Neon (not embeddings - those go to Qdrant)

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If you encounter rate limiting from Cohere:
   - Add delays between embedding requests
   - Check your Cohere plan limits
   - Consider batching requests

2. **Crawling Errors**: If some pages fail to crawl:
   - Check the base URL is correct
   - Verify network connectivity
   - Review error logs for specific page issues

3. **Qdrant Connection**: If unable to connect to Qdrant:
   - Verify URL and API key in environment variables
   - Check Qdrant Cloud account status
   - Ensure firewall allows connections to Qdrant

4. **Neon Database Connection**: If unable to connect to Neon for logging:
   - Verify connection string in environment variables
   - Check Neon database account status
   - Ensure proper permissions for logging operations

5. **Deployment Validation**: If the deployed site can't be validated:
   - Confirm the URL is accessible
   - Check for any authentication requirements
   - Verify the site structure matches expected patterns

### Testing the Installation

Run the following to verify your setup:
```bash
cd backend
python -c "import requests, bs4, cohere, qdrant_client, psycopg2; print('All dependencies installed successfully')"
```