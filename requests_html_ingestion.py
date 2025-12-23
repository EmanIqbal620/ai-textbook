import time
import random
import logging
from requests_html import HTMLSession

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def requests_html_ingest_urls(urls):
    """
    Use requests-html to render JavaScript and handle bot protection
    """
    successful_ingests = 0
    failed_ingests = 0

    session = HTMLSession()

    # Set realistic headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })

    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

            # Add random delay between requests
            delay = random.uniform(5, 15)
            logger.info(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)

            # Make the request
            response = session.get(url, timeout=30)

            if response.status_code == 200:
                # Render JavaScript content if present
                try:
                    response.html.render(timeout=20, wait=2)  # Wait for JS to execute
                except Exception as e:
                    logger.info(f"JS rendering issue for {url}, but request successful: {str(e)}")

                logger.info(f"Successfully ingested: {url} (Status: {response.status_code})")
                logger.info(f"Content length: {len(response.html.html)} characters")
                successful_ingests += 1
            elif response.status_code == 403:
                logger.warning(f"Blocked by bot protection: {url} (Status: {response.status_code})")
                failed_ingests += 1
            else:
                logger.warning(f"Failed to ingest: {url} (Status: {response.status_code})")
                failed_ingests += 1

        except Exception as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            failed_ingests += 1

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

    logger.info(f"Starting requests-html based ingestion of {len(urls)} URLs")
    logger.info("Using JavaScript rendering to handle dynamic content and bot protection")

    successful, failed = requests_html_ingest_urls(urls)

    logger.info(f"Ingestion completed.")
    logger.info(f"Results: {successful} successful, {failed} failed, out of {len(urls)} total URLs")
    print(f"\n--- SUMMARY ---")
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successfully ingested: {successful}")
    print(f"Failed attempts: {failed}")
    print(f"Success rate: {successful/len(urls)*100:.1f}%")

    return successful, failed

if __name__ == "__main__":
    main()