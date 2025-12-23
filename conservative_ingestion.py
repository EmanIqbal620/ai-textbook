import time
import requests
import random
from urllib.parse import urljoin
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def conservative_ingest_urls(urls):
    """
    Conservatively ingest URLs one by one with long delays to avoid triggering bot protection
    """
    successful_ingests = 0
    failed_ingests = 0

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    session = requests.Session()
    session.headers.update(headers)

    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

            # Add random delay between 60-90 seconds before each request
            delay = random.uniform(60, 90)
            logger.info(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)

            # Make the request
            response = session.get(url, timeout=30)

            if response.status_code == 200:
                logger.info(f"Successfully ingested: {url} (Status: {response.status_code})")
                successful_ingests += 1
            else:
                logger.warning(f"Failed to ingest: {url} (Status: {response.status_code})")
                failed_ingests += 1

                # If we get a 4xx or 5xx error, it might indicate blocking
                if response.status_code in [403, 429, 503]:
                    logger.error(f"Blocking detected (status {response.status_code}), stopping ingestion")
                    break
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            failed_ingests += 1

            # If it's a connection error or timeout, it might be blocking
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                logger.error("Connection issue detected, possibly blocked. Stopping ingestion.")
                break
        except KeyboardInterrupt:
            logger.info("Ingestion interrupted by user")
            break

    return successful_ingests, failed_ingests

def main():
    urls = [
        "https://humanoid-robotics-textbook-zeta.vercel.app/markdown-page",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/additional-resources",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/assessment-guidelines",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/bibliography",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/book-intro",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/glossary",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/hardware-requirements",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/intro",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-1-ros2/",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-1-ros2/week-1",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-1-ros2/week-2",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-1-ros2/week-3",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-2-simulation/",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-2-simulation/week-4",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-2-simulation/week-5",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-3-ai-brain/",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-3-ai-brain/week-6",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-3-ai-brain/week-7",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-3-ai-brain/week-8",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-4-vla/",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-4-vla/week-10",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-4-vla/week-11",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-4-vla/week-12",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-4-vla/week-13",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-4-vla/week-9",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-5-hardware/",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-5-hardware/hardware-specifications",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-6-assessment/",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/module-6-assessment/assessment-methods",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/prerequisites",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-basics/congratulations",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-basics/create-a-blog-post",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-basics/create-a-document",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-basics/create-a-page",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-basics/deploy-your-site",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-basics/markdown-features",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-extras/manage-docs-versions",
        "https://humanoid-robotics-textbook-zeta.vercel.app/docs/tutorial-extras/translate-your-site",
        "https://humanoid-robotics-textbook-zeta.vercel.app/"
    ]

    logger.info(f"Starting conservative ingestion of {len(urls)} URLs")
    logger.info("Using conservative approach: single requests with 60-90 second delays")
    logger.info(f"This will take approximately {len(urls) * 75 / 60:.1f} minutes to complete")

    successful, failed = conservative_ingest_urls(urls)

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