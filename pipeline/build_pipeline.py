from src.data_loader import DataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException 
from config.config import GROQ_API_KEY, MODEL_NAME as GROQ_MODEL_NAME

load_dotenv()
logger = get_logger(__name__)
def main():
    try:
        logger.info("Starting Anime Recommendation Pipeline...")
        data_loader = DataLoader(file_path="data/anime_with_synopsis.csv", processed_path="data/anime_updated.csv")
        processed_csv = data_loader.load_data()
        logger.info("Data  loaded and processed...")
        vector_store_builder = VectorStoreBuilder(processed_csv)
        logger.info("Building vector store...")
        vector_store_builder.load_and_process_data()
        logger.info("Vector store built successfully.")
    except CustomException as ce:
        logger.error(f"Custom Exception: {ce}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
    