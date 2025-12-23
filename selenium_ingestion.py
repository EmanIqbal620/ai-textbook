import time
import random
from urllib.parse import urljoin
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_selenium_driver():
    """
    Create a Chrome driver with options to appear less like a bot
    """
    chrome_options = Options()

    # Add arguments to appear more like a real user
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")  # Speed up loading
    chrome_options.add_argument("--disable-javascript")  # If you don't need JS
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # Set window size to appear more human-like
    chrome_options.add_argument("--window-size=1366,768")

    # Disable automation indicators
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    try:
        driver = webdriver.Chrome(options=chrome_options)
        # Execute script to remove webdriver property that identifies automation
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        logger.error(f"Failed to create Chrome driver: {str(e)}")
        logger.info("Make sure you have Chrome and chromedriver installed")
        logger.info("Install with: pip install selenium")
        logger.info("Download chromedriver from: https://chromedriver.chromium.org/")
        return None

def selenium_ingest_urls(urls):
    """
    Use Selenium to ingest URLs, appearing as a real browser
    """
    successful_ingests = 0
    failed_ingests = 0

    driver = create_selenium_driver()
    if not driver:
        logger.error("Cannot proceed without a working Selenium driver")
        return 0, len(urls)

    try:
        for i, url in enumerate(urls):
            try:
                logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

                # Add random delay between requests to appear more human-like
                delay = random.uniform(3, 8)
                logger.info(f"Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)

                # Navigate to the URL
                driver.get(url)

                # Wait for page to load
                try:
                    # Wait for any element to be present (indicating page loaded)
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )

                    # Get page title to confirm we got the page
                    title = driver.title
                    logger.info(f"Successfully loaded: {url} - Title: {title[:50]}...")
                    successful_ingests += 1

                except TimeoutException:
                    logger.warning(f"Timeout loading: {url}")
                    failed_ingests += 1

            except WebDriverException as e:
                logger.error(f"WebDriver error for {url}: {str(e)}")
                failed_ingests += 1
            except Exception as e:
                logger.error(f"Unexpected error for {url}: {str(e)}")
                failed_ingests += 1

    finally:
        if driver:
            driver.quit()

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

    logger.info(f"Starting Selenium-based ingestion of {len(urls)} URLs")
    logger.info("Using real browser automation to bypass bot protection")

    successful, failed = selenium_ingest_urls(urls)

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