from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt
from utils.logger import get_logger

logger = get_logger(__name__)
class AnimeRecommender:
    def __init__(self, retriever, api_key:str, model_name:str):
        # self.vector_store = vector_store
        try:
            self.llm = ChatGroq(api_key=api_key,model=model_name,temperature=0)
            self.prompt_template = get_anime_prompt()
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self.prompt_template
                },
            )
            logger.info("AnimeRecommender initialized with LLM and prompt template.")
        except Exception as e:
            logger.error(f"Failed to initialize AnimeRecommender: {e}")
            raise ValueError("Failed to initialize AnimeRecommender with the provided LLM and retriever.")

    def get_recommendation(self, question: str) -> str:
        response = self.qa_chain({"query": question})
        return response['result']