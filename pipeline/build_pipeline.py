# pipeline/build_pipeline.py (unchanged intent, just ensure it calls the new builder)
from src.data_loader import DataLoader
from src.vector_store import VectorStoreBuilder
from utils.logger import get_logger
from utils.custom_exception import CustomException
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

def main():
    try:
        logger.info("Starting Anime Recommendation Pipeline build...")
        # Create processed CSV (combined_info)
        data_loader = DataLoader(
            file_path="data/anime_with_synopsis.csv",
            processed_path="data/anime_updated.csv",
        )
        processed_csv = data_loader.load_data()
        logger.info("Data loaded and processed...")

        # Build and persist Chroma
        builder = VectorStoreBuilder(persist_dir="chroma_dir")
        builder.build_from_processed_csv(processed_csv)
        logger.info("Vector store built successfully.")
    except CustomException as ce:
        logger.error(f"Custom Exception: {ce}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
