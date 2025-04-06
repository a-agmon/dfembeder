# DfEmbedder

A high-performance Python library with a Rust core for indexing and embedding Apache Arrow compatible DataFrames (like Polars or Pandas) into a searchable Lance vector database.

## Description

DfEmbedder allows you to effortlessly turn your DataFrames into efficient vector stores. It leverages:

- **Rust:** For blazing-fast, multi-threaded embedding and indexing.
- **Apache Arrow:** To seamlessly work with data from libraries like Polars, Pandas (via PyArrow), etc.
- **Static Embeddings:** Uses efficient static models for generating vector representations.
- **Lance Format:** For optimized storage and fast vector similarity searches.
- **PyO3:** To provide a clean and easy-to-use Python API.

## Requirements

- Python (3.8+)
- PyArrow
- Polars (or Pandas, for creating DataFrames)


*Note: DfEmbedder uses the Lance file format internally. Ensure your environment can handle Lance DB creation if specific filesystem permissions or dependencies are needed.*

## Installation

### Development Setup

1.  Clone this repository
2.  Create and activate a virtual environment (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install Python dependencies
    ```bash
    pip install maturin polars pyarrow pytest
    # Add 'pandas' if you intend to use Pandas DataFrames
    # pip install pandas
    ```

### Building the Package

This command builds the Rust extension and installs the `dfembed` package into your environment.
```bash
maturin develop
```

## Usage

```python
import polars as pl
import pyarrow as pa # Although not directly used, good practice to import
from dfembed import DfEmbedder

def main():
    # --- Initialize DfEmbedder ---
    # Configure database path, vector dimensions, and optional performance params
    embedder = DfEmbedder(
        num_threads=8,              # Example: Use 8 threads for embedding
        write_buffer_size=3500,     # Example: Buffer 3500 embeddings before writing
        database_name="tmdb_db",    # Path to the Lance database directory
        vector_dim=1024             # Dimensionality of embedding vectors
    )

    # --- Load Data ---
    # Example: Load data from a CSV using Polars
    try:
        df = pl.read_csv("tmdb.csv")
    except Exception as e:
        print(f"Error reading tmdb.csv: {e}")
        print("Please ensure tmdb.csv exists in the same directory or provide the correct path.")
        return

    # --- Convert to Arrow Table ---
    # DfEmbedder requires data in PyArrow Table format
    arrow_table = df.to_arrow()

    # --- Index the Data ---
    table_name = "tmdb_table" # Name for the table within the Lance database
    print(f"Indexing data into table '{table_name}' in database '{embedder.database_name}'...")
    # This process reads the relevant column(s), generates embeddings, and saves to Lance format
    embedder.index_table(arrow_table, table_name=table_name)
    print("Indexing complete.")

    # --- Find Similar Items ---
    query = "adventures jungle animals"
    k = 10 # Number of similar results to retrieve
    print(f"\nFinding {k} items similar to: '{query}' in table '{table_name}'")

    results = embedder.find_similar(query=query, table_name=table_name, k=k)

    print("\nSimilar items found:")
    # The structure of results depends on what find_similar returns (e.g., IDs, text)
    if results:
        for i, result in enumerate(results):
            print(f"{i+1}. {result}")
    else:
        print("No similar items found.")

if __name__ == "__main__":
    main()

```

*Note: The specific column used for embedding (e.g., a movie description or plot summary in `tmdb.csv`) is determined by the internal embedding model configuration within the Rust code. Ensure your input data has the column the embedder expects.*


## Running Tests

```bash
# Ensure you are in the root directory where pyproject.toml is located
pytest python/ # Or adjust path to your tests
```


## How It Works

1.  The `DfEmbedder` Python class acts as a user-friendly wrapper.
2.  It initializes and manages an instance of the `DfEmbedderRust` struct, implemented in Rust.
3.  When `index_table` is called with a PyArrow `Table`:
    *   The Rust backend receives the Arrow data.
    *   It uses a static embedding model (configured internally) to generate vector embeddings for the specified text data, potentially using multiple threads for speed.
    *   The embeddings, along with original data, are written efficiently to a Lance dataset within the specified database directory and table name.
4.  When `find_similar` is called:
    *   The query string is embedded using the same static model.
    *   The Rust backend uses Lance's optimized search capabilities to find the `k` nearest neighbors to the query vector within the specified table.
    *   The results (e.g., identifiers or relevant data) are returned to Python.

## License

MIT