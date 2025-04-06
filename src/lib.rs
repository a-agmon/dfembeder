use embedding::static_embeder::Embedder;
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::wrap_pyfunction;
use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;

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
        info!("Initializing DfEmbedderRust");
        info!("Initializing Embedder");
        match Embedder::new() {
            Ok(embedder) => {
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
            Err(e) => {
                error!("Error initializing Embedder: {}", e);
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Error initializing Embedder: {}",
                    e
                )))
            }
        }
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
        info!("Indexing Arrow table via DfEmbedderRust");
        let py_table = convert_py_to_arrow_table(py_arrow_table)?;
        let ts = Instant::now();
        info!("Getting record batches");
        let record_batches = py_table.batches();
        info!("Got record batches in {:?}", ts.elapsed());

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
}

/// Define the Python module.
#[pymodule]
fn dfembed(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DfEmbedderRust>()?;
    Ok(())
}
