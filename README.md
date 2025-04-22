
<p align="center">
  <img src="web/main.png" alt="DF Embedder Logo" width="300"/> 

[![PyPI Downloads](https://static.pepy.tech/badge/dfembed)](https://pepy.tech/projects/dfembed) <br/>
</p>

DF Embedder is a blazing-fast Python library (with a Rust backend) that embeds, indexes and turns your dataframes into fast vector stores based on [Lance format](https://github.com/lancedb/lance) in a few lines of code.
It is aimed for use cases in which you have a dataframe with textual data that you want to embed and load to a vector db, in order to conduct vector search. It is opinionated and specifically aimed to deal with huge tables that need to be embedded fast. It's fast and efficient but uses its own embedding model and textual representation method. Read on for the details. 

*(Requires Python >= 3.10)*

```bash
pip install dfembed
```

```python
from dfembed import DfEmbedder
import polars as pl

# read a dataset using polars or pandas
df = pl.read_csv("tmdb.csv")
# turn into an arrow dataset
arrow_table = df.to_arrow()
embedder = DfEmbedder(database_name="tmdb_db")
# embed and index the dataframe to a lance table
embedder.index_table(arrow_table, table_name="films_table")
# run similarities queries
similar_movies = embedder.find_similar("adventures jungle animals", "films_table", 10)
```
See more usage examples in the notebook [here](examples/example.ipynb)

*DfEmbedder is still an early version and work in progress. Feedback and comments will be highly appriciated.*

## Main Features

- **Rust Backend:** For blazing-fast, multi-threaded embedding and indexing.
- **Apache Arrow:** To seamlessly work with data from libraries like Polars, Pandas (via PyArrow), etc.
- **Static Embeddings:** Uses efficient static embedding model for generating text embedding 100X faster.
- **Lance Format:** For optimized storage and fast vector similarity searches.
- **PyO3:** To provide a clean and easy-to-use Python API.

How fast is DF Embedder? benchamrks are often misleading and users should run their own analysis. To give a general idea, I was able to index about 1.2M rows from the TMDB movie dataset in about 100 seconds, using a machine with 10 CPU cores. Thats reading, embedding, indexing and writing more than 10K rows per second. And there are still ways to improve its performance by further tunning its params.

## How It Works

There are quite a few tabular data embedding methods (e.g. TabNet, TABBIE, etc). However, many of which assume that data has a specific structure and type whereas  RAG use cases involve embedding of unstructured free text queries into the same vector space. To resolve this I tried to follow an approach similar to the one taken by [Koloski et al.](https://arxiv.org/pdf/2502.11596) with some minor change in order to be agnostic to the field type. Accordingly, indexing a dataframe using DfEmbedder starts by representing each row in the dataframe as a string that follows the format: `col0_name is col0_value; col1_name is col1_value` (Koloski were working with a known schema and thus offered a more "typed" approach). Next, all strings are embedded using a [static embedding model](https://huggingface.co/blog/static-embeddings) (an embedding method that can generate embedding on CPU in blazing speed with very little loss of quality). Finally, it writes data as a table in Lance format.

There are several ways to search and query Lance tables created using DfEmbedder

1. You can use `DfEmbedder`'s `find_similar` method
2. You can use `LanceDB`

```python
import lancedb
db = lancedb.connect("tmdb_db")
tbl = db.open_table("films_table")
# you need the embedder to embed a query
vector = embedder.embed_string(text)
# run a vector search
tbl.search(vector).limit(10).to_list()
```

3. You can use its LlamaIndex `VectorStore` interface

```python
from dfembed import DfEmbedder, DfEmbedVectorStore

# because we use our own embedding model
Settings.embed_model = MockEmbedding(embed_dim=1024)
vector_store = DfEmbedVectorStore(
    df_embedder=embedder,
    table_name=table_name
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)

```

See more usage examples in the notebook [here](examples/example.ipynb)

## Usage

### Constructor Parameters

The `DfEmbedder` constructor accepts the following parameters:

- `num_threads` (default: CPU count): Number of parallel worker threads used for embedding.
  Setting this to the number of available CPU cores typically gives the best performance.
- `embedding_chunk_size` (default: 500): Number of records to process in each embedding batch.
  Larger values may improve throughput but require more memory.
- `write_buffer_size` (default: 2000): Number of embeddings to buffer before writing to storage.
  Increasing this reduces the number of write operations, potentially improving performance for large datasets.
- `database_name` (default: "./lance_db"): Path to the Lance database directory where tables will be stored.
- `table_name` (default: "embeddings"): Default name for tables created in the database.
  Can be overridden in `index_table()`.
- `vector_dim` (default: 1024): Dimensionality of the embedding vectors produced by the static embedder. *Please keep it on default for this version*

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

### Core Methods

- `index_table(table, table_name=None)`: Embeds and indexes an Arrow table.

  - `table`: A PyArrow Table object containing the data to index.
  - `table_name`: Name for the created Lance table. If None, uses the default name from the constructor.
- `find_similar(query, table_name, k)`: Performs semantic search for similar items.

  - `query`: String query to search for.
  - `table_name`: Name of the Lance table to search in.
  - `k`: Number of results to return.
  - Returns a list of the k most similar text records.
- `embed_string(text)`: Directly access the static embedder to encode a single string.

  - `text`: String to embed.
  - Returns a vector of floats (the embedding).

### Performance Tips

- For large datasets, increase `write_buffer_size` to reduce write operations.
- Adjust `embedding_chunk_size` based on your available memory and dataset characteristics.
- The `num_threads` parameter should typically match your CPU core count for optimal performance.
- For production use, consider using a fast SSD for the database storage location.

## License

MIT

## GitHub Actions CI/CD (WIP)

### Releasing a New Version

To release a new version:

1. Update the version in `Cargo.toml` and `pyproject.toml`
2. Commit and push your changes
3. Create and push a new tag with the format `v{version}` (e.g., `v0.1.2`):
   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```
4. The GitHub Actions workflow will automatically build wheels and publish them to PyPI

### Manual Builds

You can also manually trigger the build workflow from the GitHub Actions tab in your repository.
