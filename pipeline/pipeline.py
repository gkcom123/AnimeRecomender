from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)
class AnimeRecommendationPipeline:
    def __init__(self, processed_path: str = "chroma_dir"):
        logger.info("Initializing AnimeRecommendationPipeline...")
        vector_builder = VectorStoreBuilder(file_path="", processed_path=processed_path)
        # self.vector_store = self.vector_store_builder.load_and_process_data()
        # if not self.vector_store:
        #     raise CustomException("Failed to initialize vector store.")
        retriever = vector_builder.load_vector_store().as_retriever()

        self.recommender = AnimeRecommender(
            retriever,
            GROQ_API_KEY,
            MODEL_NAME
        )
        logger.info("AnimeRecommendationPipeline initialized successfully.")

    def recommend(self,query:str) -> str:
        try:
            logger.info(f"Recived a query {query}")
            recommendation = self.recommender.get_recommendation(query)
            logger.info("Recommendation generated sucesfulyy...")
            return recommendation
        except Exception as e:
            logger.error(f"Failed to get recommendation {str(e)}")
            raise CustomException("Error during getting recommendation" , e)
