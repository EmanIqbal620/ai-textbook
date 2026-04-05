import os
import re
import uuid
import tiktoken
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cohere  # For embeddings

class TextbookIngestionPipeline:
    def __init__(self,
                 qdrant_url: str = None,
                 qdrant_api_key: str = None,
                 cohere_api_key: str = None):
        # Initialize Qdrant client
        if qdrant_api_key:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant_client = QdrantClient(url=qdrant_url)

        # Initialize Cohere client
        self.cohere_client = cohere.Client(cohere_api_key)

        # Initialize token encoder
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Collection settings
        self.collection_name = "humanoid_ai_book"
        self.vector_size = 384  # Cohere embed-english-light-v3.0 dimension

        # Create collection if it doesn't exist
        self._setup_collection()

    def _setup_collection(self):
        """Set up Qdrant collection for textbook content"""
        try:
            # Check if collection exists
            self.qdrant_client.get_collection(self.collection_name)
            print(f"[SUCCESS] Collection {self.collection_name} already exists")
        except:
            # Create collection with cosine similarity
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"[SUCCESS] Created collection {self.collection_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"[SUCCESS] Collection {self.collection_name} already exists")
                else:
                    raise e

    def read_book_content(self, folder_path: str) -> str:
        """Read all textbook content from folder and subfolders"""
        all_content = ""

        # Walk through all subdirectories and files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".md", ".txt")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            all_content += f"\n\n--- Source: {os.path.relpath(file_path, folder_path)} ---\n\n{content}"
                            print(f"[SUCCESS] Read: {os.path.relpath(file_path, folder_path)}")
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, "r", encoding="latin-1") as f:
                                content = f.read()
                                all_content += f"\n\n--- Source: {os.path.relpath(file_path, folder_path)} ---\n\n{content}"
                                print(f"[SUCCESS] Read (latin-1): {os.path.relpath(file_path, folder_path)}")
                        except Exception as e:
                            print(f"[ERROR] Error reading {file}: {e}")
                    except Exception as e:
                        print(f"[ERROR] Error reading {file}: {e}")

        return all_content

    def clean_content(self, content: str) -> str:
        """Clean content for better embedding"""
        # Remove excessive markdown formatting
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Code blocks
        content = re.sub(r'`.*?`', '', content)  # Inline code
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Images
        content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', content)  # Links to text only
        content = re.sub(r'#+\s+', ' ', content)  # Headers
        content = re.sub(r'\*{2}|\_{2}', '', content)  # Bold
        content = re.sub(r'\*|\_', '', content)  # Italic
        content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)  # Lists

        # Normalize whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)

        return content.strip()

    def chunk_content(self, content: str, max_tokens: int = 700, overlap: int = 120) -> List[Dict]:
        """Chunk content with proper tokenization"""
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', content)

        chunks = []
        current_chunk = ""
        current_token_count = 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence_token_count = len(self.encoding.encode(sentence))

            if current_token_count + sentence_token_count <= max_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
                current_token_count += sentence_token_count
            else:
                # Add current chunk to results
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'token_count': current_token_count
                    })

                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    current_chunk_tokens = self.encoding.encode(current_chunk)
                    if len(current_chunk_tokens) > overlap:
                        overlap_tokens = current_chunk_tokens[-overlap:]
                        overlap_text = self.encoding.decode(overlap_tokens)
                        current_chunk = overlap_text + " " + sentence
                        current_token_count = len(self.encoding.encode(current_chunk))
                    else:
                        current_chunk = sentence
                        current_token_count = sentence_token_count
                else:
                    current_chunk = sentence
                    current_token_count = sentence_token_count

        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'token_count': current_token_count
            })

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere"""
        response = self.cohere_client.embed(
            texts=texts,
            model="embed-english-light-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def store_in_qdrant(self, chunks: List[Dict]):
        """Store chunks with embeddings in Qdrant"""
        if not chunks:
            print("No chunks to store!")
            return

        print(f"Storing {len(chunks)} chunks in Qdrant...")

        # Prepare points for batch upload
        points = []
        batch_size = 50  # Process in batches to avoid timeouts

        for i, chunk in enumerate(chunks):
            # Generate embedding for this chunk
            embedding = self.generate_embeddings([chunk['content']])[0]

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk['content'],
                    "token_count": chunk['token_count'],
                    "source": "textbook_content",
                    "chunk_index": i,
                    "created_at": str(uuid.uuid4())  # Simple timestamp replacement
                }
            )
            points.append(point)

            # Upload in batches
            if len(points) >= batch_size or i == len(chunks) - 1:
                self.qdrant_client.upload_points(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"[SUCCESS] Uploaded batch of {len(points)} points")
                points = []  # Reset for next batch

        print(f"[SUCCESS] Successfully stored {len(chunks)} chunks in Qdrant collection: {self.collection_name}")

    def verify_ingestion(self):
        """Verify that content was properly ingested"""
        count = self.qdrant_client.count(collection_name=self.collection_name)
        print(f"[SUCCESS] Total documents in Qdrant: {count.count}")

        # Test a sample search
        sample_query = "What is ROS 2?"
        embeddings = self.generate_embeddings([sample_query])
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embeddings[0],
            limit=3
        )

        print(f"[SUCCESS] Sample search test successful. Found {len(search_results)} results.")
        if search_results:
            print(f"Sample result content preview: {search_results[0].payload['content'][:200]}...")

    def ingest_book_folder(self, folder_path: str):
        """Complete ingestion pipeline"""
        print("Starting textbook ingestion pipeline...")

        # 1. Read content
        print("\n1. Reading textbook content...")
        raw_content = self.read_book_content(folder_path)

        if not raw_content.strip():
            print("[ERROR] No content found in the specified folder!")
            return

        print(f"[SUCCESS] Read {len(raw_content)} characters from {folder_path}")

        # 2. Clean content
        print("\n2. Cleaning content...")
        cleaned_content = self.clean_content(raw_content)
        print(f"[SUCCESS] Cleaned content: {len(cleaned_content)} characters")

        # 3. Chunk content
        print("\n3. Chunking content...")
        chunks = self.chunk_content(cleaned_content, max_tokens=700, overlap=120)
        print(f"[SUCCESS] Created {len(chunks)} chunks")

        # Show chunk statistics
        token_counts = [chunk['token_count'] for chunk in chunks]
        print(f"  Average tokens per chunk: {sum(token_counts) / len(token_counts):.1f}")
        print(f"  Max tokens per chunk: {max(token_counts)}")
        print(f"  Min tokens per chunk: {min(token_counts)}")

        # 4. Store in Qdrant
        print("\n4. Storing in Qdrant...")
        self.store_in_qdrant(chunks)

        # 5. Verify ingestion
        print("\n5. Verifying ingestion...")
        self.verify_ingestion()

        print("\n[SUCCESS] Textbook ingestion completed successfully!")

# Run the ingestion with your actual paths
if __name__ == "__main__":
    # Initialize ingestion pipeline with your credentials
    ingestion = TextbookIngestionPipeline(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    # Use the correct path to your textbook content
    book_folder_path = "D:\\Humanoid-Robotics-AI-textbook\\humanoid-robotics-textbook\\docs"

    if os.path.exists(book_folder_path):
        print(f"Found textbook content at: {book_folder_path}")
        print("Starting ingestion...")
        ingestion.ingest_book_folder(book_folder_path)
    else:
        print(f"[ERROR] Folder does not exist: {book_folder_path}")
        print("Please check the path and make sure it is correct.")