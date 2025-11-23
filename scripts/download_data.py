import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

SOURCES = {
    "langgraph_llms.txt": "https://langchain-ai.github.io/langgraph/llms.txt",
    "langgraph_llms_full.txt": "https://langchain-ai.github.io/langgraph/llms-full.txt",
    "langchain_llms.txt": "https://python.langchain.com/llms.txt",
    "langchain_v1_llms.txt": "https://docs.langchain.com/llms.txt",
    "langchain_v1_llms_full.txt": "https://docs.langchain.com/llms-full.txt"
}

def download_file(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        logger.info(f"Downloading {url} to {filepath}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info(f"Successfully downloaded {filename}")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error downloading {url}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}")
    return False

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created directory: {DATA_DIR}")

    success_count = 0
    for filename, url in SOURCES.items():
        if download_file(url, filename):
            success_count += 1
    
    logger.info(f"Download process completed. {success_count}/{len(SOURCES)} files downloaded successfully.")

if __name__ == "__main__":
    main()

