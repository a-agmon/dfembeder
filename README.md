# DF Embedder

DF Embedder allows you to effortlessly turn your DataFrames into fast vector stores in 3 lines of code. 

## Description

DF Embedder is a high-performance Python library (with a Rust backend) for indexing and embedding Apache Arrow compatible DataFrames (like Polars or Pandas) into low latency vector databases based on Lance files.

- **Rust:** For blazing-fast, multi-threaded embedding and indexing.
- **Apache Arrow:** To seamlessly work with data from libraries like Polars, Pandas (via PyArrow), etc.
- **Static Embeddings:** Uses efficient static embedding model for generating text embedding 100X faster.
- **Lance Format:** For optimized storage and fast vector similarity searches.
- **PyO3:** To provide a clean and easy-to-use Python API.

## Usage

```python
import polars as pl # could also use Pandas or DuckDB
import pyarrow as pa # Although not directly used, good practice to import
from dfembed import DfEmbedder

def main():   
    # Example: Load data from a CSV using Polars
    df = pl.read_csv("tmdb.csv")
    # DfEmbedder requires data in PyArrow Table format
    arrow_table = df.to_arrow()
     # Configure database path, vector dimensions, and optional performance params
    embedder = DfEmbedder(
        num_threads=8,              # Example: Use 8 threads for embedding or defaults the cores available
        write_buffer_size=3500,     # Example: Buffer 3500 embeddings before writing
        database_name="tmdb_db",    # Path to the Lance database directory            
    )
    table_name = "tmdb_table" # Name for the table within the Lance database
    # This process embeds and indexes all rows, and saves to Lance format
    embedder.index_table(arrow_table, table_name=table_name)

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