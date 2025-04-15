import os
import pyarrow as pa
# Import the Rust class from the compiled extension
from .dfembed import DfEmbedderRust

class DfEmbedder:
    """
    Python wrapper class for managing the Rust DfEmbedderRust instance.
    Provides a high-level interface with Python defaults and type hints.
    """
    def __init__(
        self,
        num_threads=os.cpu_count(),
        embedding_chunk_size=500,
        write_buffer_size=2000,
        database_name="./lance_db",
        table_name="embeddings", # Default table name for convenience
        vector_dim=1024,
    ):
        """
        Initialize the DfEmbedder, creating an instance of the Rust backend class.

        Args:
            num_threads: Number of parallel threads for embedding (default: number of CPU cores).
            embedding_chunk_size: Number of records to process in each embedding batch (default: 500).
            write_buffer_size: Number of embeddings to buffer before writing to storage (default: 2000).
            database_name: Path to the Lance database directory (default: "./lance_db").
            table_name: Default name for the Lance table within the database (default: "embeddings").
                      This can be overridden in the `index_table` method.
            vector_dim: Dimensionality of the embedding vectors (default: 1024).
        """
        # Store Python-side config/defaults
        self.database_name = database_name
        self.default_table_name = table_name # Store the default table name

        # Create and store the Rust backend instance
        # Note: table_name is no longer passed to the Rust constructor
        self._rust_embedder = DfEmbedderRust(
            num_threads=num_threads,
            embedding_chunk_size=embedding_chunk_size,
            write_buffer_size=write_buffer_size,
            database_name=database_name,
            # table_name=table_name, # Removed
            vector_dim=vector_dim,
        )

    def index_table(self, table: pa.Table, table_name: str | None = None):
        """
        Index an Arrow table using the configured Rust backend.

        Args:
            table: PyArrow Table object to index.
            table_name: Name of the table to create/update in the database.
                        If None, uses the default table name provided during initialization.

        Returns:
            None. Raises an exception on Rust error.
        """
        if not isinstance(table, pa.Table):
            raise TypeError("Input must be a PyArrow Table object.")

        # Determine the table name to use
        target_table_name = table_name if table_name is not None else self.default_table_name
        if not target_table_name:
             raise ValueError("Table name must be provided either during initialization or in the index_table call.")

        # Delegate the call to the Rust instance method, passing the table_name
        self._rust_embedder.index_table(table, target_table_name)

    def analyze_table(self, table: pa.Table, detailed=False):
        """
        Analyze an Arrow table structure using the Rust backend.

        Args:
            table: PyArrow Table object
            detailed: Whether to show detailed analysis (default: False).
                      Note: Currently ignored by the Rust backend.

        Returns:
            None. Raises an exception on Rust error.
        """
        if not isinstance(table, pa.Table):
            raise TypeError("Input must be a PyArrow Table object.")

        # Delegate the call to the Rust instance method
        # The detailed parameter is currently ignored
        self._rust_embedder.analyze_table(table)

    def find_similar(self, query: str, table_name: str, k: int):
        """
        Find similar items in the database using the configured Rust backend.

        Args:
            query: The query string to find similar items for.
            table_name: The name of the table to search in.
            k: The number of similar items to return.

        Returns:
            A list of the k most similar items to the query.
        """
        return self._rust_embedder.find_similar(query, table_name, k)

    def embed_string(self, text: str) -> list[float]:
        """
        Embed a single string using the static embedding model.

        Args:
            text: The string to embed.

        Returns:
            A list of floats representing the embedding of the string.
        """
        return self._rust_embedder.embed_string(text)

    # Example of accessing configuration (if needed)
    # def get_database_path(self) -> str:
    #    # If the Rust class exposed its fields (needs #[pyo3(get)])
    #    # return self._rust_embedder.database_path
    #    # Otherwise, return the value stored during Python init
    #    return self.database_name

    # We could add methods here later to interact with persistent state
    # held by self._rust_embedder, like closing connections, etc. 