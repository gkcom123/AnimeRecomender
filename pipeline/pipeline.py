# pipeline/pipeline.py
import os
from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self, processed_path: str = "chroma_dir"):
        logger.info("Initializing AnimeRecommendationPipeline...")
        builder = VectorStoreBuilder(persist_dir=processed_path)

        # Fail fast if the store isn't built
        vector_store = builder.load_or_raise()
        retriever = builder.get_retriever(k=4)

        self.recommender = AnimeRecommender(
            retriever=retriever,
            api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
        )
        logger.info("AnimeRecommendationPipeline initialized successfully.")

    def recommend(self, query: str) -> str:
        try:
            logger.info(f"Received query: {query}")
            recommendation = self.recommender.get_recommendation(query)
            logger.info("Recommendation generated successfully.")
            return recommendation
        except Exception as e:
            logger.error(f"Failed to get recommendation: {str(e)}")
            raise CustomException("Error during getting recommendation", e)
