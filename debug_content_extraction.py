import requests
from bs4 import BeautifulSoup

def debug_content_extraction():
    """Debug content extraction by examining the actual HTML structure"""
    print("Debugging content extraction...")

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
        print("Total div count:", len(soup.find_all('div')))
        print("Total p count:", len(soup.find_all('p')))
        print("Total article count:", len(soup.find_all('article')))

        # Look for script tags that might contain the content
        scripts = soup.find_all('script')
        print(f"Script tags found: {len(scripts)}")

        # Look for content in main tags
        main_tags = soup.find_all('main')
        print(f"Main tags found: {len(main_tags)}")
        for i, main_tag in enumerate(main_tags):
            content = main_tag.get_text(strip=True)
            print(f"Main tag {i} content length: {len(content)}")
            if content:
                print(f"  Sample: {content[:100]}...")

        # Look for content in divs with specific classes
        divs = soup.find_all('div')
        print("\nAnalyzing divs with content...")
        content_divs = []
        for div in divs:
            text_content = div.get_text(strip=True)
            if len(text_content) > 100:  # Only consider divs with substantial content
                classes = div.get('class', [])
                content_divs.append((len(text_content), classes, text_content[:100]))

        # Sort by content length and show top 5
        content_divs.sort(key=lambda x: x[0], reverse=True)
        print(f"Top {min(5, len(content_divs))} content divs:")
        for i, (length, classes, sample) in enumerate(content_divs[:5]):
            print(f"  {i+1}. Length: {length}, Classes: {classes}")
            print(f"     Sample: {sample}...")

        # Try the extraction logic from the original script
        print("\nTesting original extraction logic:")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", ".sidebar"]):
            script.decompose()

        # Try to find main content areas
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
                print(f"Selector '{selector}' found content: {len(content)} characters")
                if len(content) > len(text_content):
                    text_content = content
                    print(f"  Sample: {content[:100]}...")
            else:
                print(f"Selector '{selector}' found no elements")

        print(f"\nFinal extracted content length: {len(text_content)}")
        if text_content:
            print(f"Sample: {text_content[:200]}...")

    except Exception as e:
        print(f"Error in debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_content_extraction()