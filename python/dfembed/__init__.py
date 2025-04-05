"""
dfembed - A Python library using Rust to index Arrow tables.
"""

import os

# Import functions from the underlying Rust extension module
from .dfembed import index_arrow_table, analyze_arrow_table

__version__ = "0.1.0"

# Python wrapper functions with different signatures
def index_table(
    table,
    num_threads=os.cpu_count(),
    embedding_chunk_size=500,
    write_buffer_size=2000,
    database_name="./lance_db",
    table_name="embeddings",
    vector_dim=1024, # Default to typical embedding model dimension
):
    """
    Index an Arrow table using the Rust backend with customizable parameters.

    Args:
        table: PyArrow Table object to index.
        num_threads: Number of parallel threads for embedding (default: number of CPU cores).
        embedding_chunk_size: Number of records to process in each embedding batch (default: 500).
        write_buffer_size: Number of embeddings to buffer before writing to storage (default: 2000).
        database_name: Path to the Lance database directory (default: "./lance_db").
        table_name: Name of the Lance table within the database (default: "embeddings").
        vector_dim: Dimensionality of the embedding vectors (default: 1024).

    Returns:
        None
    """
    # Pass the parameters to the underlying Rust function
    return index_arrow_table(
        table,
        num_threads,
        embedding_chunk_size,
        write_buffer_size,
        database_name,
        table_name,
        vector_dim,
    )

def analyze_table(table, detailed=False):
    """
    Analyze an Arrow table structure.
    
    Args:
        table: PyArrow Table object
        detailed: Whether to show detailed analysis (default: False)
    
    Returns:
        None
    """
    # The detailed parameter could be passed to the Rust function
    # if we modify it to accept this parameter
    return analyze_arrow_table(table)

def quick_index(table):
    """
    Quick index an Arrow table with default parameters.

    Args:
        table: PyArrow Table object

    Returns:
        None
    """
    # Call the main index_table function with its default parameters
    return index_table(table)

# Define what 'from dfembed import *' imports
__all__ = ["index_arrow_table", "analyze_arrow_table", 
           "index_table", "analyze_table"]