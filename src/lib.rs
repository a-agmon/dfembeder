use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::wrap_pyfunction;
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

/// Analyzes an Arrow table by printing its schema and processing record batches.
/// Analyzes an Arrow table by printing its schema and processing record batches.
///
/// This function receives a PyArrow table from Python, converts it to Apache Arrow record batches,
/// and analyzes each batch, printing schema and batch information.
#[pyfunction]
fn analyze_arrow_table(py_arrow_table: &Bound<'_, PyAny>) -> PyResult<()> {
    init_tracing();
    info!("Analyzing Arrow table");
    // Convert PyArrow table to Arrow table
    let py_table = convert_py_to_arrow_table(py_arrow_table)?;

    // Get record batches
    let ts = Instant::now();
    info!("Getting record batches");
    let record_batches = py_table.batches();
    info!("Got record batches in {:?}", ts.elapsed());

    if record_batches.is_empty() {
        info!("Arrow Table contains no batches.");
        return Ok(());
    }

    // Get schema from the first batch
    let schema = record_batches[0].schema();

    // Print schema
    print_schema(&schema);

    Ok(())
}

#[pyfunction]
fn index_arrow_table(
    py_arrow_table: &Bound<'_, PyAny>,
    num_threads: usize,
    chunk_size: usize,
    write_buffer_size: usize,
    database_name: &str,
    table_name: &str,
    vector_dim: usize,
) -> PyResult<()> {
    init_tracing();
    info!("Indexing Arrow table");
    // Convert PyArrow table to Arrow table
    let py_table = convert_py_to_arrow_table(py_arrow_table)?;
    // Get record batches
    let ts = Instant::now();
    info!("Getting record batches");
    let record_batches = py_table.batches();
    info!("Got record batches in {:?}", ts.elapsed());
    if record_batches.is_empty() {
        error!("Arrow Table contains no batches.");
        return Ok(());
    }
    // Get schema from the first batch
    let schema = record_batches[0].schema();
    let indexer = Indexer::new(record_batches, schema);
    let result = indexer.run(
        num_threads,
        chunk_size,
        write_buffer_size,
        database_name,
        table_name,
        vector_dim,
    );
    if let Err(e) = result {
        error!("Error indexing arrow table: {}", e);
    }

    Ok(())
}

/// Define the Python module. The function `analyze_arrow_table` will be exposed to Python.
#[pymodule]
fn dfembed(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_arrow_table, m)?)?;
    m.add_function(wrap_pyfunction!(index_arrow_table, m)?)?;
    Ok(())
}
