import time
import requests
import random
from urllib.parse import urljoin
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_session_with_retries():
    """
    Create a requests session with retry strategy and proper headers
    """
    session = requests.Session()

    # Define retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,  # Exponential backoff
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set realistic browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }

    session.headers.update(headers)
    return session

def sophisticated_ingest_urls(urls):
    """
    Ingest URLs using more sophisticated techniques to avoid bot detection
    """
    successful_ingests = 0
    failed_ingests = 0

    session = create_session_with_retries()

    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

            # Add random delay between 5-15 seconds for this more sophisticated approach
            # (shorter than the previous conservative approach since we're using better techniques)
            delay = random.uniform(5, 15)
            logger.info(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)

            # Make the request with additional measures
            response = session.get(
                url,
                timeout=30,
                # Add referer header to appear more like a real browser navigation
                headers={
                    'Referer': 'https://www.google.com/' if i == 0 else urls[i-1]
                }
            )

            if response.status_code == 200:
                logger.info(f"Successfully ingested: {url} (Status: {response.status_code})")
                logger.info(f"Content length: {len(response.content)} bytes")
                successful_ingests += 1
            elif response.status_code == 403:
                logger.warning(f"Blocked by bot protection: {url} (Status: {response.status_code})")
                logger.info("This site has strong bot protection. Consider using a browser automation tool like Selenium.")
                failed_ingests += 1
            else:
                logger.warning(f"Failed to ingest: {url} (Status: {response.status_code})")
                failed_ingests += 1

                # If we get a 4xx or 5xx error, it might indicate blocking
                if response.status_code in [403, 429, 503]:
                    logger.error(f"Blocking detected (status {response.status_code}), continuing with next URL")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            failed_ingests += 1

            # If it's a connection error or timeout, it might be blocking
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                logger.error("Connection issue detected, possibly blocked.")

        except KeyboardInterrupt:
            logger.info("Ingestion interrupted by user")
            break

    return successful_ingests, failed_ingests

def main():
    urls = [
        "https://humanoid-robotics-textbook-4ufa.vercel.app/markdown-page",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/additional-resources",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/assessment-guidelines",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/bibliography",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/book-intro",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/glossary",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/hardware-requirements",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/intro",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-1-ros2/",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-1-ros2/week-1",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-1-ros2/week-2",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-1-ros2/week-3",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-2-simulation/",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-2-simulation/week-4",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-2-simulation/week-5",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-3-ai-brain/",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-3-ai-brain/week-6",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-3-ai-brain/week-7",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-3-ai-brain/week-8",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-4-vla/",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-4-vla/week-10",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-4-vla/week-11",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-4-vla/week-12",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-4-vla/week-13",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-4-vla/week-9",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-5-hardware/",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-5-hardware/hardware-specifications",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-6-assessment/",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/module-6-assessment/assessment-methods",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/prerequisites",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-basics/congratulations",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-basics/create-a-blog-post",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-basics/create-a-document",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-basics/create-a-page",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-basics/deploy-your-site",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-basics/markdown-features",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-extras/manage-docs-versions",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/docs/tutorial-extras/translate-your-site",
        "https://humanoid-robotics-textbook-4ufa.vercel.app/"
    ]

    logger.info(f"Starting sophisticated ingestion of {len(urls)} URLs")
    logger.info("Using enhanced approach: realistic headers, retry logic, and browser-like behavior")

    successful, failed = sophisticated_ingest_urls(urls)

    logger.info(f"Ingestion completed.")
    logger.info(f"Results: {successful} successful, {failed} failed, out of {len(urls)} total URLs")
    print(f"\n--- SUMMARY ---")
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successfully ingested: {successful}")
    print(f"Failed attempts: {failed}")
    print(f"Success rate: {successful/len(urls)*100:.1f}%")

    if failed > 0 and successful == 1:
        print(f"\n--- RECOMMENDATION ---")
        print(f"The site has strong bot protection. Consider using one of these alternatives:")
        print(f"1. Use Selenium with a real browser: pip install selenium")
        print(f"2. Use requests-html which can render JavaScript: pip install requests-html")
        print(f"3. Use a headless browser approach")
        print(f"4. Check if the site has an API or RSS feed available")

    return successful, failed

if __name__ == "__main__":
    main()