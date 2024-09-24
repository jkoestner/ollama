"""Generic Helper Functions for LLM."""

import datetime
import json
import os
import shutil
from pathlib import Path

import tiktoken
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from osllmh.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

ROOT_PATH = Path(__file__).resolve().parent.parent
OSLLMH_INPUTS_PATH = (
    Path(os.getenv("OSLLMH_INPUTS_PATH"))
    if os.getenv("OSLLMH_INPUTS_PATH")
    else ROOT_PATH
)


class Engine:
    """
    Engine is a wrapper for llama index.

    Example usage:
        import osllmh
        e = osllmh.engine.Engine()
        response = e.query("Your question here")
        e.update_index()
        e.delete_index()
    """

    def __init__(self, files_dir=None, storage_dir=None, settings=True):
        """
        Initialize the engine.

        Parameters
        ----------
        files_dir : str
            Directory where the documents are stored.
        storage_dir : str
            Directory where the index will be persisted.
        settings : bool
            Whether to load settings from a file.

        """
        # initiate the directories
        if files_dir is None:
            self.files_dir = os.path.join(OSLLMH_INPUTS_PATH, "files")
        else:
            self.files_dir = files_dir
        if storage_dir is None:
            self.storage_dir = os.path.join(OSLLMH_INPUTS_PATH, "storage")
        else:
            self.storage_dir = storage_dir
        if settings:
            self.load_settings()

        self.index_dir = os.path.join(self.storage_dir, "index")

        self.log_file_path = os.path.join(self.storage_dir, "queries.log")
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w") as f:
                f.write("--- Query Log ---\n")

        # initiate the variables
        self.index = None
        self.query_engine = None
        self.query_log = []

        # create the engine
        self.token_counter = self._setup_tokenizer()
        self.create_index()
        self.create_query_engine()

    def create_index(self):
        """
        Create the index.

        Returns
        -------
        index : VectorStoreIndex
            The index object.

        """
        if not os.path.exists(self.index_dir):
            logger.info("No existing index found. Creating a new index...")
            # Load documents and create a new index
            documents = SimpleDirectoryReader(self.files_dir).load_data()
            self.index = VectorStoreIndex.from_documents(documents)
            # Persist the index for future use
            self.index.storage_context.persist(persist_dir=self.index_dir)
            logger.info("Index created and persisted.")
        else:
            logger.info("Loading index from storage...")
            # Load the existing index from storage
            storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            self.index = load_index_from_storage(storage_context)
            logger.info("Index loaded from storage.")

    def create_query_engine(self, **kwargs):
        """
        Create the query engine from the index.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to the query engine.
               - similarity_top_k : int

        Returns
        -------
        query_engine : QueryEngine
            The query engine object.

        """
        self.query_engine = self.index.as_query_engine(**kwargs)

    def query(self, question):
        """
        Query the index with the given question.

        Parameters
        ----------
        question : str
            The question to query the index with.

        Returns
        -------
        repsonse : str
            The response from the query engine.

        """
        response = self.query_engine.query(question)
        token_usage = self.get_token_counts()

        # log the query
        self._log_query(question, response.response, token_usage)

        return response

    def update_index(self, new_files_dir=None):
        """
        Update the index with new documents.

        Parameters
        ----------
        new_files_dir : str (optional)
            New directory containing documents.

        """
        if new_files_dir:
            update_dir = new_files_dir
        else:
            update_dir = self.files_dir
        logger.info(f"Updating index with new documents from {update_dir}...")
        documents = SimpleDirectoryReader(update_dir).load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.index.storage_context.persist(persist_dir=self.index_dir)
        logger.info("Index updated and persisted.")

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

    def get_settings(self, save=False):
        """
        Get the current settings of the engine.

        to change settings, use the Settings class from llama_index and
        update the values.

        reference:
        https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/

        Parameters
        ----------
        save : bool
            Whether to save the settings to directory.

        Returns
        -------
        settings : dict
            The settings of the engine.

        """
        settings = {
            "llm": {
                "model": Settings.llm.model,
                "temperature": Settings.llm.temperature,
            },
            "embed_model": {
                "model_name": Settings.embed_model.model_name,
                "embed_batch_size": Settings.embed_model.embed_batch_size,
            },
            "text_splitter": {
                "chunk_size": Settings.text_splitter.chunk_size,
                "chunk_overlap": Settings.text_splitter.chunk_overlap,
            },
            "prompt_helper": {
                "context_window": Settings.context_window,
                "num_output": Settings.num_output,
            },
        }

        if save:
            output_dir = os.path.join(self.storage_dir, "settings.json")
            logger.info(f"Settings saved to {output_dir}")
            with open(output_dir, "w") as f:
                json.dump(settings, f, indent=4)

        return settings

    def load_settings(self, settings_path=None):
        """
        Load settings from a file.

        Parameters
        ----------
        settings_path : str or dict (optional)
            The file or dictionary containing the settings.
            If none, assumes the settings file is in the persist directory named
            'settings.json'.

        Returns
        -------
        settings : dict
            The settings loaded from the file.

        """
        if settings_path is None:
            settings_path = os.path.join(self.storage_dir, "settings.json")

        # check if settings path is path or dictionary
        if isinstance(settings_path, dict):
            settings = settings_path
        else:
            # check if the file exists
            if not os.path.exists(settings_path):
                logger.warning(
                    f"Settings file not found at {settings_path}, run "
                    f"without 'settings' = 'True' if this is the first time"
                )
                return
            # load the file
            logger.info("Loading settings...")
            with open(settings_path, "r") as f:
                settings = json.load(f)

        # update the settings
        Settings.llm.model = settings["llm"]["model"]
        Settings.llm.temperature = settings["llm"]["temperature"]
        Settings.embed_model.model_name = settings["embed_model"]["model_name"]
        Settings.embed_model.embed_batch_size = settings["embed_model"][
            "embed_batch_size"
        ]
        Settings.text_splitter.chunk_size = settings["text_splitter"]["chunk_size"]
        Settings.context_window = settings["prompt_helper"]["context_window"]
        Settings.num_output = settings["prompt_helper"]["num_output"]

        return settings

    def reset_tokenizer(self):
        """Reset the tokenizer for the engine."""
        self.token_counter.reset_counts()

    def get_token_counts(self):
        """
        Get the token counts from the tokenizer.

        Returns
        -------
        token_counts : dict
            The token counts from the tokenizer.

        """
        token_counts = {
            "embedding_tokens": self.token_counter.total_embedding_token_count,
            "llm_prompt_tokens": self.token_counter.prompt_llm_token_count,
            "llm_completion_tokens": self.token_counter.completion_llm_token_count,
            "total_tokens": self.token_counter.total_llm_token_count,
        }
        return token_counts

    def _setup_tokenizer(self):
        """Set up the tokenizer for the engine."""
        logger.info("Setting up the tokenizer...")
        token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(Settings.llm.model).encode
        )
        Settings.callback_manager = CallbackManager([token_counter])

        return token_counter

    def _log_query(self, query, response, token_usage):
        """
        log the query to a file.

        Parameters
        ----------
        query : str
            The query that was made.
        response : str
            The response from the query engine.
        token_usage : dict
            The token usage from the query.

        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response,
            "token_usage": token_usage,
        }
        with open(self.log_file_path, "a") as log_file:
            log_file.write(f"Timestamp: {log_entry['timestamp']}\n")
            log_file.write(f"Query: {log_entry['query']}\n")
            log_file.write(f"Response: {log_entry['response']}\n")
            log_file.write(f"Token Usage: {log_entry['token_usage']}\n")
            log_file.write("\n---\n")  # Separator between entries


# Example usage
if __name__ == "__main__":
    # Initialize the query engine
    engine = Engine()

    # Query the index
    question = "What did the author do growing up?"
    response = engine.query(question)
    print("Question:", question)
    print("Response:", response)
