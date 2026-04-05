import requests
from bs4 import BeautifulSoup
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_ingestion_logic():
    """Test the complete ingestion logic to see where it's failing"""
    print("Testing complete ingestion logic...")

    url = "https://humanoid-robotics-textbook-zeta.vercel.app/docs/intro"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements (same as in the original script)
        for script in soup(["script", "style", "nav", "header", "footer", ".sidebar"]):
            script.decompose()

        # Try to find main content areas (same as in the original script)
        content_selectors = [
            '.theme-doc-markdown', '.markdown', 'article', '.main-wrapper',
            '.doc-content', '.docs-content', '.main-content', 'main', '.container',
            '[class*="docItemContainer"]', '[class*="docRoot"]', '[class*="docItemCol"]'
        ]

        text_content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator=' ', strip=True)
                if len(content) > len(text_content):
                    text_content = content

        print(f"Extracted text length: {len(text_content)}")
        if text_content:
            print(f"Sample: {text_content[:200]}...")
        else:
            print("No text extracted")
            return

        # Now test the chunking logic
        print("\nTesting chunking logic...")

        # Import the token encoding (simplified version)
        # Instead of using tiktoken, let's use a simple approach for testing
        # This is similar to what the original script does
        max_tokens = 500  # Same as original script

        # Split text into sentences/paragraphs (same as original)
        sentences = text_content.split('\n')
        if len(sentences) == 1:  # If no newlines, split by sentences
            sentences = [s.strip() for s in text_content.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n').split('\n') if s.strip()]

        print(f"Split into {len(sentences)} sentences/paragraphs")

        # Create chunks (same logic as original)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Estimate token count - using character count as a rough proxy
            # In the original script, it uses tiktoken but we'll use a simpler approach for testing
            chunk_with_sentence = current_chunk + " " + sentence if current_chunk else sentence
            rough_token_count = len(chunk_with_sentence) // 4  # Rough approximation

            if rough_token_count <= max_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:  # If current chunk is not empty, save it
                    chunks.append(current_chunk.strip())
                # Start a new chunk with the current sentence
                current_chunk = sentence

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"Created {len(chunks)} chunks from text")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1} length: {len(chunk)} chars")
            print(f"  Sample: {chunk[:100]}...")

        if not chunks:
            print("ERROR: No chunks created - this would cause the ingestion to fail!")
        else:
            print("SUCCESS: Chunks were created successfully")

    except Exception as e:
        print(f"Error in ingestion logic test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ingestion_logic()