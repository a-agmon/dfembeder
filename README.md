# DF Embedder

DF Embedder allows you to effortlessly turn your DataFrames into fast vector stores in 3 lines of code. 

```python
df = pl.read_csv("tmdb.csv")
arrow_table = df.to_arrow()
embedder = DfEmbedder(database_name="tmdb_db")
```

## Description

DF Embedder is a high-performance Python library (with a Rust backend) for indexing and embedding Apache Arrow compatible DataFrames (like Polars or Pandas) into low latency vector databases based on Lance files.

- **Rust:** For blazing-fast, multi-threaded embedding and indexing.
- **Apache Arrow:** To seamlessly work with data from libraries like Polars, Pandas (via PyArrow), etc.
- **Static Embeddings:** Uses efficient static embedding model for generating text embedding 100X faster.
- **Lance Format:** For optimized storage and fast vector similarity searches.
- **PyO3:** To provide a clean and easy-to-use Python API.

How fast is DF Embedder? benchamrks are often misleading and users should run their own analysis. To give a general idea, I was able to index about 1.2M rows from the TMDB movie dataset in about 120 seconds. Thats reading, embedding, indexing and writing more than 10K rows per second. And there are still ways to improve its performance.  

## Usage

```python
import polars as pl # could also use Pandas or DuckDB
import pyarrow as pa # Although not directly used, good practice to import
from dfembed import DfEmbedder

# Load data from a CSV using Polars
df = pl.read_csv("tmdb.csv")
# transform to PyArrow Table format
arrow_table = df.to_arrow()
# Configure database path, and optional performance params
embedder = DfEmbedder(
    num_threads=8,              # Use 8 threads for embedding or defaults to avail num of cores
    write_buffer_size=3500,     # Buffer 3500 embeddings before writing
    database_name="tmdb_db",    # Path to the Lance database directory            
)
table_name = "tmdb_table" 
embedder.index_table(arrow_table, table_name=table_name)
# get 10 most similar items
query = "adventures jungle animals"
results = embedder.find_similar(query=query, table_name=table_name, k=10)

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