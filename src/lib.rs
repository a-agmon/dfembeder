use embedding::static_embeder::Embedder;
use once_cell::sync::Lazy;
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tracing::debug;
use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;
use storage::lance::LanceStore;
use tokio::runtime::Runtime;

mod arrow;
use arrow::utils::{convert_py_to_arrow_table, print_schema};
mod embedding;
mod storage;

use tracing::error;
use tracing::info;
mod indexer;
use indexer::Indexer;

// Static Once variable to ensure initialization happens only once
static INIT: Once = Once::new();

// Initialize tracing once
fn init_tracing() {
    INIT.call_once(|| {
        tracing_subscriber::fmt::init();
    });
}

// Global Tokio runtime instance
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime")
});

#[pyclass(module = "dfembed.dfembed")]
struct DfEmbedderRust {
    num_threads: usize,
    embedding_chunk_size: usize,
    write_buffer_size: usize,
    database_path: PathBuf,
    vector_dim: usize,
    embedder: Embedder,
}

#[pymethods]
impl DfEmbedderRust {
    #[new]
    #[pyo3(signature = (
        num_threads,
        embedding_chunk_size,
        write_buffer_size,
        database_name,
        vector_dim
    ))]
    fn new(
        num_threads: usize,
        embedding_chunk_size: usize,
        write_buffer_size: usize,
        database_name: String,
        vector_dim: usize,
    ) -> PyResult<Self> {
        init_tracing();
        info!("Initializing Embedder");
        let embedder = Embedder::new().expect("Failed to create real Embedder");
        info!("Embedder initialized");
        Ok(DfEmbedderRust {
            num_threads,
            embedding_chunk_size,
            write_buffer_size,
            database_path: PathBuf::from(database_name),
            vector_dim,
            embedder,
        })
    }

    /// Analyzes an Arrow table by printing its schema.
    fn analyze_table(&self, py_arrow_table: &Bound<'_, PyAny>) -> PyResult<()> {
        info!("Analyzing Arrow table via DfEmbedderRust");
        let py_table = convert_py_to_arrow_table(py_arrow_table)?;
        let record_batches = py_table.batches();
        if record_batches.is_empty() {
            info!("Arrow Table contains no batches.");
            return Ok(());
        }

        let schema = record_batches[0].schema();
        print_schema(&schema);

        Ok(())
    }

    /// Indexes an Arrow table using the configuration stored in the DfEmbedderRust instance.
    fn index_table(&self, py_arrow_table: &Bound<'_, PyAny>, table_name: &str) -> PyResult<()> {
        debug!("Indexing Arrow table via DfEmbedderRust");
        let py_table = convert_py_to_arrow_table(py_arrow_table)?;
        let ts = Instant::now();
        debug!("Getting record batches");
        let record_batches = py_table.batches();
        debug!("Got record batches in {:?}", ts.elapsed());

        if record_batches.is_empty() {
            error!("Arrow Table contains no batches.");
            return Ok(());
        }

        let schema = record_batches[0].schema();
        let indexer = Indexer::new(record_batches, schema);

        let result = indexer.run(
            self.num_threads,
            self.embedding_chunk_size,
            self.write_buffer_size,
            &self.database_path.to_string_lossy(),
            table_name,
            self.vector_dim,
        );
        if let Err(e) = result {
            error!("Error indexing arrow table: {}", e);
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Error indexing arrow table: {}",
                e
            )));
        }

        Ok(())
    }

    /// Finds similar items to a query vector in the specified table.
    /// Blocks until the search completes and returns a Vec<String>.
    fn find_similar(&self, query: String, table_name: String, k: usize) -> PyResult<Vec<String>> {
        let db_path = self.database_path.clone();
        let vector_dim = self.vector_dim;
        // Assuming Embedder doesn't need cloning if find_most_similar takes &Embedder
        // If it does need Clone, add: let embedder = self.embedder.clone();
        let embedder_ref = &self.embedder; // Pass a reference

        RUNTIME.block_on(async move {
            let vector_store =
                LanceStore::new_with_database(&db_path.to_string_lossy(), &table_name, vector_dim);
            vector_store
                .find_most_similar(&query, k, embedder_ref) // Use reference
                .await
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Error finding similar items: {}",
                        e
                    ))
                })
        })
    }

    /// Embeds a single string using the static embedding model.
    fn embed_string(&self, text: &str) -> PyResult<Vec<f32>> {
        let text_vec = vec![text];
        let result = self.embedder.embed_batch_vec(&text_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error embedding string: {}",
                e
            ))
        })?;
        // asert that we have one embedding
        assert_eq!(result.len(), 1);
        Ok(result[0].clone())
    }
}

/// Define the Python module.
#[pymodule]
fn dfembed(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DfEmbedderRust>()?;
    Ok(())
}
