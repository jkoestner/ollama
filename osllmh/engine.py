"""
Generic Helper Functions for LLM.

The process is to:
  - Create an Engine object.
    - The engine object is initiated with parameters
  - Update the index with new documents by using a vector store and
    creates storage context.
    - The documents are read from a directory with a number of parameters
  - Use a response synthesizer to create a response from the index.
    - the response synthesizer has prompt templates
  - Query the index with the response synthesizer.
    - The index are retrieved from a vector store
    - The response is then post-processed

"""

import datetime
import os
from pathlib import Path

import qdrant_client
import tiktoken
import yaml
from IPython.display import Markdown, display
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    constants,
    get_response_synthesizer,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.vector_stores.qdrant import QdrantVectorStore

from osllmh import vector_stores
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
        e.create_index()
        e.delete_index()
    """

    def __init__(self):
        """Initialize the engine."""
        # initiate the variables
        self.index = None
        self.query_engine = None

        # initiate the directories
        # storage
        self.storage_dir = os.path.join(OSLLMH_INPUTS_PATH, "storage")
        if not os.path.exists(self.storage_dir):
            raise FileNotFoundError(f"Files directory not found at {self.storage_dir}")
        # log
        self.log_file_path = os.path.join(self.storage_dir, "queries.log")
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w") as f:
                f.write("--- Query Log ---\n")
        # settings
        self.settings = self.update_settings(settings_path=None)
        self._check_settings()

        # create the index and query engine
        self.token_counter = self._setup_tokenizer()
        self.create_or_load_index()

    def create_or_load_index(self):
        """
        Create the index.

        by default the vectorstore is a simple in memory store.

        Returns
        -------
        index : VectorStoreIndex
            The index object.

        """
        # create index
        if not self.vector_provider.check_index_exists():
            self.create_index(files_dir=self.files_dir)
        # load index
        else:
            logger.info("Loading existing index from storage...")
            self.index = self.vector_provider.load_index()
            self.create_query_engine()

    def create_index(self, files_dir=None):
        """
        Update the index with new documents, supporting recursive directory traversal.

        Parameters
        ----------
        files_dir : str (optional)
            New directory containing documents.

        """
        update_dir = files_dir or self.files_dir

        logger.info(f"Updating index with new documents from {update_dir}...")
        documents = SimpleDirectoryReader(update_dir, recursive=True).load_data()
        unique_files = set()

        # create new index if doesn't exist
        if not self.vector_provider.check_index_exists():
            logger.info(
                f"No existing index found. Creating a new index "
                f"with `{self.vector_type}`..."
            )
            for document in documents:
                doc_file_path = document.metadata.get("file_path", None)
                if doc_file_path not in unique_files:
                    unique_files.add(doc_file_path)
            logger.info(f"Found {len(unique_files)} new documents.")
            self.index = self.vector_provider.create_index(documents)
        # adding to index
        else:
            logger.info(
                f"Loading existing index from storage with " f"{self.vector_type}`..."
            )
            self.index = self.vector_provider.load_index()
            existing_files = self.vector_provider.list_files_from_index()
            existing_file_paths = {
                file_info["file_path"] for file_info in existing_files
            }

            # filter out documents that are already in the index
            new_documents = []
            for document in documents:
                doc_file_path = document.metadata.get("file_path", None)
                if doc_file_path and doc_file_path not in existing_file_paths:
                    new_documents.append(document)
                    if doc_file_path not in unique_files:
                        unique_files.add(doc_file_path)

            logger.info(f"Adding {len(unique_files)} new documents to the index...")
            self.index = self.vector_provider.update_index(self.index, new_documents)

        # Persist the updated index for future use
        self.vector_provider.persist_index(self.index)
        self.create_query_engine()

        return self.index

    def create_response_synthesizer(self, **kwargs):
        """
        Create the response synthesizer.

        link:
          - https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to the query engine.
              - response_mode : str
              The response mode to use.
              default = "compact"
              other options include: "refine", "tree_summarize"

        Returns
        -------
        response_synthesizer : ResponseSynthesizer
            The response synthesizer object.

        """
        if kwargs is None:
            kwargs = {}

        # set the option parameters
        if "response_mode" not in kwargs:
            kwargs["response_mode"] = self.settings["prompt_helper"]["response_mode"]

        # create the response synthesizer
        logger.info(
            f"Creating response synthesizer with "
            f"reponse_mode: {kwargs['response_mode']}..."
        )
        self.response_synthesizer = get_response_synthesizer(**kwargs)

        return self.response_synthesizer

    def create_query_engine(self, prompt_section=None, **kwargs):
        """
        Create the query engine from the index.

        link:
          - https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/constants.py
          - https://docs.llamaindex.ai/en/stable/api_reference/
          - https://docs.llamaindex.ai/en/stable/api_reference/retrievers/vector/#llama_index.core.retrievers.VectorIndexRetriever.similarity_top_k
          - https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/
          - https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/
          - https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/


        Parameters
        ----------
        prompt_section : str, optional
            The section of the prompt to use in the yaml
            (e.g. "text_qa_template", "refine_template").
        kwargs : dict
            Additional keyword arguments to pass to the query engine.
              - similarity_top_k : int
                default = 2
                how many similar nodes to return
              - node_postprocessors : list
                default = None
                any further processing to nodes such as filtering

        Returns
        -------
        query_engine : QueryEngine
            The query engine object.

        """
        if kwargs is None:
            kwargs = {}
        if prompt_section is None:
            prompt_section = self.settings["prompt_helper"].get("prompt_section", None)
        self.create_response_synthesizer()

        # set the option parameters
        if "similarity_top_k" not in kwargs:
            kwargs["similarity_top_k"] = self.settings["engine"]["nodes_similar"]
        if "node_postprocessors" not in kwargs:
            kwargs["node_postprocessors"] = None

        # create the query engine
        logger.info("Creating query engine...")
        self.query_engine = self.index.as_query_engine(
            response_synthesizer=self.response_synthesizer, **kwargs
        )

        # creating a custom prompt
        if prompt_section is not None:
            self.update_prompts(prompt_section)

        return self.query_engine

    def query(self, question, reset=True):
        """
        Query the index with the given question.

        Parameters
        ----------
        question : str
            The question to query the index with.
        reset : bool (optional)
            Whether to reset the tokenizer token counts.

        Returns
        -------
        query_response : QueryResponse
            The response from the query engine.

        """
        response = self.query_engine.query(question)
        token_usage = self.get_token_counts()

        # log the query
        self._log_query(question, response.response, token_usage)

        query_response = QueryResponse(
            full_response=response,
            token_usage=token_usage,
        )

        if reset:
            self.reset_tokenizer()

        return query_response

    def get_settings(self, save=False):
        """
        Get the current settings of the engine.

        To change settings, use the Settings class from llama_index and
        update the values.

        Reference:
        https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/

        Parameters
        ----------
        save : bool
            Whether to save the settings to a directory.

        Returns
        -------
        settings : dict
            The settings of the engine.

        """
        # check user settings
        node_similar = self.settings["engine"].get(
            "nodes_similar", constants.DEFAULT_SIMILARITY_TOP_K
        )
        response_mode = self.settings["prompt_helper"].get("response_mode", "compact")
        prompt_section = self.settings["prompt_helper"].get("prompt_section", "compact")
        project_name = self.settings["project"].get("name", "default")
        vector_type = self.settings["project"].get("vector_type", "base")
        qdrant_url = self.settings["project"].get("qdrant_url", None)

        # get the package settings
        settings = {
            "project": {
                "name": project_name,
                "vector_type": vector_type,
                "qdrant_url": qdrant_url,
            },
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
                "response_mode": response_mode,
                "prompt_section": prompt_section,
            },
            "engine": {"nodes_similar": node_similar},
        }
        self._update_directories(settings)

        if save:
            output_dir = os.path.join(self.storage_dir, "settings.yml")
            logger.info(f"Settings saved to {output_dir}")
            with open(output_dir, "w") as f:
                yaml.dump(settings, f, default_flow_style=False)

        return settings

    def update_settings(self, settings_path=None, recreate_index=False):
        """
        Load settings from a file.

        Parameters
        ----------
        settings_path : str or dict (optional)
            The file or dictionary containing the settings.
            If none, assumes the settings file is in the persist directory named
            'settings.yml'.
        recreate_index : bool (optional)
            Whether to recreate the index after loading the settings.

        Returns
        -------
        settings : dict
            The settings loaded from the file.

        """
        if settings_path is None:
            settings_path = os.path.join(self.storage_dir, "settings.yml")

        # check if settings path is path or dictionary
        if isinstance(settings_path, dict):
            settings = settings_path
        else:
            # check if the file exists
            if not os.path.exists(settings_path):
                logger.warning(
                    f"Settings file not found at {settings_path}. "
                    f"Run without 'settings' = 'True' if this is the first time."
                )
                return
            # load the file
            with open(settings_path, "r") as f:
                settings = yaml.safe_load(f)

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

        self.settings = settings
        self._update_directories(settings)
        self.project_name = self.settings["project"].get("name", "default")
        self.vector_type = self.settings["project"].get("vector_type", "base")
        qdrant_url = self.settings["project"].get("qdrant_url", None)
        has_url = ", and url" if qdrant_url else None
        self.vector_provider = vector_stores.VectorStore(
            vector_type=self.vector_type,
            index_dir=self.index_dir,
            collection_name=self.project_name,
            url=qdrant_url,
        ).vector_provider
        self.vector_store = self.vector_provider.vector_store

        logger.info(
            f"Loaded settings with project: `{self.project_name}`, "
            f"vector_type: `{self.vector_type}`{has_url}..."
        )

        # new settings will trigger a need to recreate the index and query engine
        if recreate_index:
            self.create_or_load_index()
        elif self.index is not None:
            self.create_query_engine()

        return settings

    def display_prompt_dict(self):
        """Display the prompts from the response synthesizer."""
        prompts_dict = self.query_engine.get_prompts()
        for k, p in prompts_dict.items():
            text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
            display(Markdown(text_md))
            print(p.get_template())
            display(Markdown("<br><br>"))

    def update_prompts(self, prompt_section):
        """
        Update the prompts from the 'prompts.yml'.

        Parameters
        ----------
        prompt_section : str
            The section of the prompt to use in the yaml
            (e.g. "refine", "compact").

        """
        prompt_path = os.path.join(self.storage_dir, "prompts.yml")
        # checks
        if not os.path.exists(prompt_path):
            logger.warning(
                f"Prompt file not found at {prompt_path}. " f"Ensure file is available."
            )
            return None

        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)
        if prompt_section not in prompts:
            logger.warning(
                f"Prompt section '{prompt_section}' not found. "
                f"Ensure prompt section and prompt are available."
            )
            return None

        # loop through the prompts
        logger.info(f"Using custom prompt: {prompt_section}...")
        for prompt in prompts[prompt_section]:
            prompt_tmpl_str = prompts[prompt_section][prompt]
            prompt_tmpl = PromptTemplate(prompt_tmpl_str)
            self.query_engine.update_prompts(
                {f"response_synthesizer:{prompt}": prompt_tmpl}
            )

    def reset_tokenizer(self):
        """Reset the tokenizer for the engine."""
        self.token_counter.reset_counts()

    def get_token_counts(self, reset=True):
        """
        Get the token counts from the tokenizer.

        Parameters
        ----------
        reset : bool (optional)
            Whether to reset the tokenizer token counts.

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

        if reset:
            self.token_counter.reset_counts()

        return token_counts

    def _check_settings(self):
        """Check the current settings of the engine."""
        settings = self.get_settings()
        needed_settings = [
            "project",
            "llm",
            "embed_model",
            "text_splitter",
            "prompt_helper",
            "engine",
        ]
        missing_settings = [key for key in needed_settings if key not in settings]
        if missing_settings:
            raise ValueError(
                f"Settings missing: {missing_settings}. Please update the settings."
            )

    def _get_vector_store(self, vector_type="base", collection_name=None):
        """
        Get the vector store for the index.

        Parameters
        ----------
        vector_type : str (optional)
            The type of vector store to use.
        collection_name : str (optional)
            The name of the collection to use (qdrant only).

        Returns
        -------
        vector_store : object
            The vector store object.

        """
        if vector_type == "base":
            vector_store = None
        elif vector_type == "qdrant":
            client = qdrant_client.QdrantClient(location=":memory:")
            vector_store = QdrantVectorStore(
                client=client, collection_name=collection_name
            )
        else:
            raise ValueError("Vector store not supported.")

        return vector_store

    def _update_directories(self, settings):
        """
        Update the directories for the engine.

        Parameters
        ----------
        settings : dict
            The settings of the engine.

        Returns
        -------
        files_dir : str
            The updated files directory.
        index_dir : str
            The updated index directory

        """
        self.project_name = settings["project"]["name"]
        # files
        self.files_dir = os.path.join(OSLLMH_INPUTS_PATH, "files", self.project_name)
        if not os.path.exists(self.files_dir):
            raise FileNotFoundError(f"Files directory not found at {self.files_dir}")
        # index
        self.index_dir = os.path.join(self.storage_dir, self.project_name)
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"Files directory not found at {self.index_dir}")

        return self.files_dir, self.index_dir

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
        # read
        with open(self.log_file_path, "r") as log_file:
            existing_log = log_file.readlines()

        max_lines = 10000
        if len(existing_log) > max_lines:
            existing_log = existing_log[:max_lines]

        # create log
        new_log = (
            f"Timestamp: {datetime.datetime.now().isoformat()}\n"
            f"Query: {query}\n"
            f"Response: {response}\n"
            f"Token Usage: {token_usage}\n"
            f"Response Mode: {self.settings['prompt_helper']['response_mode']}\n"
            f"Prompt Section: {self.settings['prompt_helper']['prompt_section']}\n"
            "---\n"  # Separator between entries
        )

        updated_log = new_log + "".join(existing_log)

        # prepend
        with open(self.log_file_path, "w") as log_file:
            log_file.write(updated_log)


class QueryResponse:
    """QueryResponse is a wrapper for the response from the query engine."""

    def __init__(self, full_response, token_usage=None):
        """
        Initialize the QueryResponse object with attributes.

        Parameters
        ----------
        full_response : object
            The full response object returned by the query engine.
        token_usage : dict
            The token usage from the query.

        """
        self.response = full_response.response
        self.meta = self.response_meta(full_response)
        self.token_usage = token_usage
        self.full_response = full_response

    def __repr__(self):
        """Return a string representation of the QueryResponse object."""
        return (
            f"QueryResponse"
            f"(response='{self.response[:30]}...', "
            f"meta={len(self.meta)} items)"
        )

    def response_meta(self, response):
        """
        Get the metadata of the response.

        Parameters
        ----------
        response : str
            The response from the query engine.

        Returns
        -------
        meta : list
            The metadata of the response.

        """
        meta = []
        for node in response.source_nodes:
            item = {
                "source": node.node.metadata.get("file_name", "Unknown file"),
                "ref_doc_id": node.node.ref_doc_id,
                "score": node.score,
                # assuming 3000 characters per page
                "approx_page": node.node.start_char_idx / 3000,
                "page": node.node.metadata.get("page_label", None),
                "start_char": node.node.start_char_idx,
                "text": node.node.text[:200],
                "break": "----------------",
            }
            meta.append(item)

        logger.info(f"metadata for {len(meta)} sources")

        return meta

    def get_node(self, ref_doc_id=None, node_idx=None):
        """
        Get the node from the response.

        Parameters
        ----------
        ref_doc_id : str
            The ref_doc_id to retrieve.
        node_idx : int
            The index of the item to retrieve.

        Returns
        -------
        node : object
            The node object from the response.

        """
        nodes = self.full_response.source_nodes

        if (ref_doc_id is None and node_idx is None) or (ref_doc_id and node_idx):
            logger.warning("Please provide either ref_doc_id or node_idx.")
            return None
        elif ref_doc_id and not node_idx:
            for node in nodes:
                # Check if the node id matches the target_node_id
                if node.node.ref_doc_id == ref_doc_id:
                    return node.node
        elif node_idx and not ref_doc_id:
            return nodes[node_idx].node

        return None
