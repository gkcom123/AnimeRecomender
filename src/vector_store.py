from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from utils.logger import get_logger
load_dotenv()
logger = get_logger(__name__)

class VectorStoreBuilder:
    def __init__(self, file_path: str, processed_path: str = "chroma_dir"):
        self.file_path = file_path
        self.processed_path = processed_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(embedding_function=self.embeddings)

    def load_and_process_data(self):
        try:
            loader = CSVLoader(file_path=self.file_path, 
                               encoding='utf-8', 
                               metadata_columns=[]  )
            data = loader.load()
            if not data:
                raise ValueError("No data found in the file.")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(data)
            logger.info(f"Number of documents loaded: {len(texts)}")

            db = Chroma.from_documents(
                texts, 
                self.embeddings, 
                persist_directory=self.processed_path
            )
            logger.info("Vector store created successfully.")
            # db.persist()
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    def load_vector_store(self):
        try:
            self.vector_store = Chroma(persist_directory=self.processed_path, 
                                       embedding_function=self.embeddings)
            return self.vector_store
        except Exception as e:
            print(f"An error occurred while loading the vector store: {e}")
            return None