# src/vector_store.py

import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()
logger = get_logger(__name__)


class VectorStoreBuilder:
    """
    Responsible for building, persisting, and loading a Chroma vector store.

    Typical use:
        builder = VectorStoreBuilder(persist_dir="chroma_dir")
        # Build once (offline) from processed CSV:
        builder.build_from_processed_csv("data/anime_updated.csv")

        # Later at runtime:
        vs = builder.load_or_raise()
        retriever = builder.get_retriever(k=4)
    """

    def __init__(
        self,
        persist_dir: str = "chroma_dir",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "anime",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._vector_store: Optional[Chroma] = None

    # ---------- Public API ----------

    def build_from_processed_csv(
        self,
        processed_csv_path: str,
        content_column: Optional[str] = None,
        encoding: str = "utf-8",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> Chroma:
        """
        Build the vector store from a *processed* CSV (e.g., `data/anime_updated.csv`)
        that already contains the combined text to embed.

        If `content_column` is None, CSVLoader will use default behavior (often first column).
        """
        try:
            self._ensure_file_exists(processed_csv_path)

            logger.info(f"Loading CSV from {processed_csv_path}")
            loader = CSVLoader(
                file_path=processed_csv_path,
                encoding=encoding,
                csv_args={"quotechar": '"'},
                source_column=content_column,  # can be None; CSVLoader will choose default
            )
            docs: List[Document] = loader.load()

            if not docs:
                raise CustomException(
                    f"No documents loaded from {processed_csv_path}. "
                    f"Check the CSV and `content_column`."
                )

            logger.info(f"Splitting {len(docs)} doc(s) into chunks")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(docs)
            if not chunks:
                raise CustomException("Text splitting produced no chunks.")

            logger.info(f"Creating Chroma collection at {self.persist_dir}")
            vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )

            logger.info(f"Adding {len(chunks)} chunk(s) to vector store")
            vs.add_documents(chunks)
            self._vector_store = vs

            # quick sanity check
            if self._is_empty(vs):
                raise CustomException(
                    "Vector store appears empty after build. "
                    "Verify your CSV content and columns."
                )

            logger.info("Vector store built and persisted successfully.")
            return vs

        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise CustomException("Error while building vector store", e)

    def load(self) -> Optional[Chroma]:
        """
        Attempt to load an existing vector store. Returns None if missing/empty.
        """
        try:
            if not os.path.isdir(self.persist_dir):
                logger.warning(f"Persist dir not found: {self.persist_dir}")
                return None

            vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )

            if self._is_empty(vs):
                logger.warning(
                    f"Chroma collection at {self.persist_dir} is empty (no vectors)."
                )
                return None

            self._vector_store = vs
            logger.info(f"Loaded vector store from {self.persist_dir}")
            return vs

        except Exception as e:
            logger.error(f"Error while loading vector store: {e}")
            return None

    def load_or_raise(self) -> Chroma:
        """
        Load the vector store or raise a helpful error instructing to build first.
        """
        vs = self.load()
        if vs is None:
            raise CustomException(
                f"Vector store not available at '{self.persist_dir}'. "
                f"Run the build step first (e.g., pipeline/build_pipeline.py)."
            )
        return vs

    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """
        Return a retriever from the loaded vector store.
        """
        if self._vector_store is None:
            # try loading lazily
            self.load_or_raise()
        return self._vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )

    # ---------- Helpers ----------

    @staticmethod
    def _ensure_file_exists(path: str) -> None:
        if not os.path.isfile(path):
            raise CustomException(f"File not found: {path}")

    @staticmethod
    def _is_empty(vs: Chroma) -> bool:
        """
        Heuristic check: if the collection returns no IDs on a small get(), treat as empty.
        """
        try:
            data = vs.get(limit=1)
            ids = data.get("ids", [])
            return len(ids) == 0
        except Exception:
            # if get() fails, assume empty/broken
            return True
