"""Modulize to normalize the vector stores."""

import os
import shutil

import qdrant_client
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models as rest
from tqdm.auto import tqdm

from osllmh.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


class VectorStore:
    """A wrapper class to handle different types of vector stores."""

    def __init__(self, vector_type, **kwargs):
        if vector_type == "qdrant":
            self.vector_provider = QdrantVS(**kwargs)
        elif vector_type == "base":
            self.vector_provider = BaseVS(index_dir=kwargs.get("index_dir"))


class BaseVS:
    """A base vector store with basic file management methods."""

    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.vector_store = None

    def check_index_exists(self):
        """Check if an index exists in the storage."""
        exists_docstore = os.path.exists(os.path.join(self.index_dir, "docstore.json"))
        exists_vector = os.path.exists(
            os.path.join(self.index_dir, "default__vector_store.json")
        )
        exists = exists_docstore and exists_vector
        return exists

    def list_files_from_index(self):
        """
        List the documents stored in the index.

        Returns
        -------
        files_info : list
            A list of file names stored in the index.

        """
        storage_context = self.index.storage_context
        document_store = storage_context.docstore
        all_documents = document_store.docs

        seen_files = set()
        files_info = []

        for doc_id, doc in all_documents.items():
            file_name = doc.metadata.get("file_name", f"Unknown file ({doc_id})")
            file_path = doc.metadata.get("file_path", "Path not available")

            file_identifier = (file_name, file_path)

            if file_identifier not in seen_files:
                files_info.append({"file_name": file_name, "file_path": file_path})
                seen_files.add(file_identifier)

        return files_info

    def create_index(self, documents):
        """
        Create an index from the documents.

        Parameters
        ----------
        documents : list
            A list of documents to create the index from.

        Returns
        -------
        index : VectorStoreIndex
            The created index.

        """
        # create index with first document
        first_document = [documents[0]]
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents=first_document,
            storage_context=storage_context,
        )

        # update index with remaining documents
        if len(documents) > 1:
            self.index = self.update_index(index=self.index, documents=documents[1:])

        return self.index

    def update_index(self, index, documents):
        """
        Update the index with new documents.

        Parameters
        ----------
        index : VectorStoreIndex
            The index to update.
        documents : list
            A list of documents to update the index with.

        Returns
        -------
        index : VectorStoreIndex
            The updated index.

        """
        with tqdm(total=len(documents), desc="Indexing documents") as pbar:
            for document in documents:
                index.insert(document)
                pbar.update(1)
        self.index = index

        return self.index

    def load_index(self):
        """
        Load an index from a file.

        Returns
        -------
        index : VectorStoreIndex
            The loaded index.

        """
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=self.index_dir,
        )
        self.index = load_index_from_storage(storage_context)

        return self.index

    def delete_index(self):
        """Delete the persisted index from storage."""
        if os.path.exists(self.index_dir):
            logger.info("Deleting the index from storage...")

            shutil.rmtree(self.index_dir)
            os.makedirs(self.index_dir)
            self.index = None
            self.query_engine = None
            logger.info("Index deleted.")
        else:
            logger.info("No index found to delete.")

    def persist_index(self, index):
        """
        Persist the index to a file.

        Parameters
        ----------
        index : VectorStoreIndex
            The index to persist.

        """
        logger.info("Index updated and persisted.")
        index.storage_context.persist(persist_dir=self.index_dir)


class QdrantVS:
    """A Qdrant-based vector store."""

    def __init__(self, collection_name, index_dir, url=None):
        self.url = url
        self.index_dir = index_dir
        if self.url:
            self.client = qdrant_client.QdrantClient(url=url)
        else:
            self.client = qdrant_client.QdrantClient(path=index_dir)
        self.collection_name = collection_name
        self.vector_store = QdrantVectorStore(
            client=self.client, collection_name=self.collection_name
        )

    def check_index_exists(self):
        """Check if an index exists in the storage."""
        docstore_exists = os.path.exists(os.path.join(self.index_dir, "docstore.json"))
        exists_vector = self.client.collection_exists(
            collection_name=self.collection_name
        )
        exists = docstore_exists and exists_vector

        return exists

    def list_files_from_index(self):
        """List the documents stored in the index."""
        scroll_position = None
        files_info = []
        file_paths = []

        while True:
            result, scroll_position = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,  # Number of documents to fetch per scroll
                offset=scroll_position,
            )

            # collect file paths from the documents
            for point in result:
                file_path = point.payload.get("file_path")
                if file_path and file_path not in file_paths:
                    file_paths.append(file_path)
                    doc_id = point.payload.get("doc_id")
                    files_info.append({"file_path": file_path, "doc_id": doc_id})

            if not scroll_position:
                break

        return files_info

    def create_index(self, documents):
        """
        Create an index from the documents.

        Parameters
        ----------
        documents : list
            A list of documents to create the index from.

        Returns
        -------
        index : VectorStoreIndex
            The created index.

        """
        # create index with first document
        first_document = [documents[0]]
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents=first_document,
            storage_context=storage_context,
        )

        # update index with remaining documents
        if len(documents) > 1:
            self.index = self.update_index(index=self.index, documents=documents[1:])

        return self.index

    def update_index(self, index, documents):
        """
        Update the index with new documents.

        Parameters
        ----------
        index : VectorStoreIndex
            The index to update.
        documents : list
            A list of documents to update the index with.

        Returns
        -------
        index : VectorStoreIndex
            The updated index.

        """
        with tqdm(total=len(documents), desc="Indexing documents") as pbar:
            for document in documents:
                index.insert(document)
                pbar.update(1)
        self.index = index

        return self.index

    def load_index(self):
        """
        Load an index from a file.

        Returns
        -------
        index : VectorStoreIndex
            The loaded index.

        """
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=self.index_dir,
        )
        self.index = load_index_from_storage(storage_context)

        return self.index

    def delete_index(self):
        """Delete the persisted index from storage."""
        if os.path.exists(self.index_dir):
            logger.info("Deleting the index from storage...")

            shutil.rmtree(self.index_dir)
            os.makedirs(self.index_dir)
            self.index = None
            self.query_engine = None
            if self.url:
                self.client.delete_collection(collection_name=self.collection_name)
            logger.info("Index deleted.")
        else:
            logger.info("No index found to delete.")

    def delete_document(self, doc_id):
        """
        Delete a document from the index.

        Parameters
        ----------
        doc_id : str
            The ID of the document to delete.

        """
        logger.info(f"Deleting document with doc_id: {doc_id}")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=doc_id)
                    )
                ]
            ),
        )

    def persist_index(self, index):
        """
        Persist the index to a file.

        Parameters
        ----------
        index : VectorStoreIndex
            The index to persist.

        """
        logger.info("Index updated and persisted.")
        index.storage_context.persist(persist_dir=self.index_dir)
