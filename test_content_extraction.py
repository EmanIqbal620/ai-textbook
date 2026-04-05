import requests
from bs4 import BeautifulSoup

def test_content_extraction():
    """Test content extraction from a sample textbook page"""
    print("Testing content extraction from textbook page...")

    # Test with a sample page from the textbook
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

        print("Page title:", soup.title.string if soup.title else "No title")

        # Try different selectors that are common in Docusaurus
        selectors_to_try = [
            'article',
            '.theme-doc-markdown',
            '.markdown',
            '.doc-content',
            '.docs-content',
            '.main-wrapper',
            '.container',
            '.theme-content',
            '[class*="docItemContainer"]',
            '[class*="markdown"]',
            'main'
        ]

        print("\nTrying different selectors:")
        for selector in selectors_to_try:
            elements = soup.select(selector)
            if elements:
                content = elements[0].get_text(separator=' ', strip=True)
                if len(content) > 50:  # Only show if there's meaningful content
                    print(f"Selector '{selector}': {len(content)} characters")
                    print(f"  Sample: {content[:100]}...")
                    print()
            else:
                print(f"Selector '{selector}': No elements found")

        # Also check for common Docusaurus class patterns
        print("\nChecking for Docusaurus-specific classes:")
        all_divs = soup.find_all('div')
        docusaurus_classes = []
        for div in all_divs:
            if div.get('class'):
                classes = div.get('class')
                for cls in classes:
                    if 'doc' in cls.lower() or 'markdown' in cls.lower() or 'theme' in cls.lower():
                        if cls not in docusaurus_classes:
                            docusaurus_classes.append(cls)

        print("Potential Docusaurus classes found:", docusaurus_classes)

    except Exception as e:
        print(f"Error extracting content: {e}")

if __name__ == "__main__":
    test_content_extraction()