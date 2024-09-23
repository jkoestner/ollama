"""Generic Helper Functions for LLM."""

import json
import logging
import os
import sys
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
from llama_index.llms.openai import OpenAI

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

ROOT_PATH = Path(__file__).resolve().parent.parent


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

    def __init__(self, data_dir=None, persist_dir=None):
        """
        Initialize the engine.

        Parameters
        ----------
        data_dir : str
            Directory where the documents are stored.
        persist_dir : str
            Directory where the index will be persisted.

        """
        # initiate the directories
        if data_dir is None:
            self.data_dir = os.path.join(ROOT_PATH, "data")
        else:
            self.data_dir = data_dir
        if persist_dir is None:
            self.persist_dir = os.path.join(ROOT_PATH, "storage")
        else:
            self.persist_dir = persist_dir
        self.index = None
        self.query_engine = None
        self.token_counter = self._setup_tokenizer()
        self._load_or_create_index()

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
        if self.query_engine is None:
            self._load_or_create_index()
        response = self.query_engine.query(question)
        return response

    def update_index(self, new_data_dir=None):
        """
        Update the index with new documents.

        Parameters
        ----------
        new_data_dir : str (optional)
            New directory containing documents.

        """
        if new_data_dir:
            self.data_dir = new_data_dir
        logging.info("Updating index with new documents from '%s'...", self.data_dir)
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        self.query_engine = self.index.as_query_engine()
        logging.info("Index updated and persisted.")

    def delete_index(self):
        """Delete the persisted index from storage."""
        if os.path.exists(self.persist_dir):
            logging.info("Deleting the index from storage...")
            import shutil

            shutil.rmtree(self.persist_dir)
            self.index = None
            self.query_engine = None
            logging.info("Index deleted.")
        else:
            logging.info("No index found to delete.")

    def save_settings(self, output=True):
        """
        Get the current settings of the engine.

        to change settings, use the Settings class from llama_index and
        update the values.

        reference:
        https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/

        Parameters
        ----------
        output : bool
            Whether to output the settings to directory.

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
            "text_splitter": {"chunk_size": Settings.text_splitter.chunk_size},
            "prompt_helper": {
                "context_window": Settings.context_window,
                "num_output": Settings.num_output,
            },
        }

        if output:
            output_dir = os.path.join(self.persist_dir, "settings.json")
            logging.info(f"Settings saved to {output_dir}")
            with open(output_dir, "w") as f:
                json.dump(settings, f, indent=4)

        return settings

    def load_settings(self, settings_file=None):
        """
        Load settings from a file.

        Parameters
        ----------
        settings_file : str (optional)
            The file containing the settings.
            If none, assumes the settings file is in the persist directory named
            'settings.json'.

        """
        if settings_file is None:
            settings_file = os.path.join(self.persist_dir, "settings.json")
        if not os.path.exists(settings_file):
            raise FileNotFoundError(f"Settings file not found at {settings_file}")

        # load the file
        with open(settings_file, "r") as f:
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

        logging.info(f"Settings loaded from {settings_file}")

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

    def _load_or_create_index(self):
        """
        Load and create the index if it doesn't exist.

        Returns
        -------
        index : VectorStoreIndex
            The index object.

        """
        if not os.path.exists(self.persist_dir):
            logging.info("No existing index found. Creating a new index...")
            # Load documents and create a new index
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            self.index = VectorStoreIndex.from_documents(documents)
            # Persist the index for future use
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            logging.info("Index created and persisted.")
        else:
            logging.info("Loading index from storage...")
            # Load the existing index from storage
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            logging.info("Index loaded from storage.")

        self.query_engine = self.index.as_query_engine()

    def _setup_tokenizer(self):
        """Set up the tokenizer for the engine."""
        logging.info("Setting up the tokenizer...")
        token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(Settings.llm.model).encode
        )
        Settings.callback_manager = CallbackManager([token_counter])

        return token_counter


# Example usage
if __name__ == "__main__":
    # Initialize the query engine
    engine = Engine()

    # Query the index
    question = "What did the author do growing up?"
    response = engine.query(question)
    print("Question:", question)
    print("Response:", response)
