# DF Embedder

DF Embedder is a high-performance Python library (with a Rust backend) that embeds, indexes and turns your dataframes into fast vector stores (based on [Lance format](https://github.com/lancedb/lance)) in a few lines of code.

```python
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

*DfEmbedder is an early version and work in progress. Feedback and comments will be highly appriciated.*

## Main Features

- **Rust Backend:** For blazing-fast, multi-threaded embedding and indexing.
- **Apache Arrow:** To seamlessly work with data from libraries like Polars, Pandas (via PyArrow), etc.
- **Static Embeddings:** Uses efficient static embedding model for generating text embedding 100X faster.
- **Lance Format:** For optimized storage and fast vector similarity searches.
- **PyO3:** To provide a clean and easy-to-use Python API.

How fast is DF Embedder? benchamrks are often misleading and users should run their own analysis. To give a general idea, I was able to index about 1.2M rows from the TMDB movie dataset in about 100 seconds, using a machine with 10 CPU cores. Thats reading, embedding, indexing and writing more than 10K rows per second. And there are still ways to improve its performance by further tunning its params.

## How does DF Embedder work?

Indexing a dataframe using DfEmbedder starts by representing each row in the dataframe as a string that follows the format: `col0_name is col0_value; col1_name is col1_value`. Next, all strings are embedded using a [static embedding model](https://huggingface.co/blog/static-embeddings) (an embedding method that can generate embedding on CPU in blazing speed with very little loss of quality). Finally, it writes data as a table in Lance format.

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

3. You can use its Llamaindex `VectorStore` interface

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

1. The `DfEmbedder` Python class acts as a user-friendly wrapper.
2. It initializes and manages an instance of the `DfEmbedderRust` struct, implemented in Rust.
3. When `index_table` is called with a PyArrow `Table`:
   * The Rust backend receives the Arrow data.
   * It uses a static embedding model (configured internally) to generate vector embeddings for the specified text data, potentially using multiple threads for speed.
   * The embeddings, along with original data, are written efficiently to a Lance dataset within the specified database directory and table name.
4. When `find_similar` is called:
   * The query string is embedded using the same static model.
   * The Rust backend uses Lance's optimized search capabilities to find the `k` nearest neighbors to the query vector within the specified table.
   * The results (e.g., identifiers or relevant data) are returned to Python.

## License

MIT

## GitHub Actions CI/CD

This project uses GitHub Actions for continuous integration and deployment, automatically building wheels for multiple platforms and Python versions.

### Automated Builds

The CI/CD pipeline automatically:

1. Builds wheels for:

   - Linux (manylinux2014)
   - macOS (Intel x86_64 and Apple Silicon ARM64)
   - Windows
   - Python versions 3.8, 3.9, 3.10, 3.11, and 3.12
2. Tests the built wheels on each platform to ensure they work correctly
3. Publishes to PyPI when a new tag is pushed (format: `v*`, e.g., `v0.1.2`)

### Workflow Files

- `.github/workflows/build.yml`: Builds wheels for all platforms and Python versions
- `.github/workflows/test.yml`: Tests the built wheels to ensure they work correctly

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
