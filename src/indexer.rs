use arrow::array::RecordBatch;
use arrow::datatypes::Schema;

use std::sync::Arc;

use crossbeam::channel;
use crossbeam::channel::Sender;
use tokio::runtime::Runtime;
use tracing::error;
use tracing::info;

use crate::embedding::coordinator::EmbeddingCoordinator;
use crate::embedding::static_embeder::Embedder;
use crate::storage::lance::LanceStore;
use crate::storage::lance_writer;

pub struct Indexer {
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
}

impl Indexer {
    pub fn new(batches: &[RecordBatch], schema: Arc<Schema>) -> Self {
        let _ = Embedder::new().unwrap();
        Self {
            batches: batches.to_vec(),
            schema,
        }
    }

    /// This function orchestrates the main workflow:
    /// 1. Transforms Arrow record batches into text chunks
    /// 2. Spawns  embedding worker threads that:
    ///    - Receive text chunks from a channel
    ///    - Generate embeddings using the static embedding model
    ///    - Send results to a writer channel
    /// 3. Runs a writer thread that stores the embeddings and metadata in a Lance database
    pub fn run(
        &self,
        num_workers: usize,
        embedding_chunk_size: usize,
        write_buffer_size: usize,
        database_name: &str,
        table_name: &str,
        vector_dim: usize,
    ) -> anyhow::Result<()> {
        info!(
            "Starting indexer with {} workers and embedding chunk size {} and write buffer size {}",
            num_workers, embedding_chunk_size, write_buffer_size
        );
        // Initialize Tokio runtime for the writer thread
        let rt = Arc::new(Runtime::new()?);
        let (send_to_embedder, receive_from_embedder) = channel::unbounded();
        let (send_to_writer, receive_from_writer) = channel::unbounded();
        let store = LanceStore::new_with_database(database_name, table_name, vector_dim);
        let store = Arc::new(store);

        // transform the batches to text chunks and send them to the embedder
        if let Err(e) = transform_batches(&self.batches, &self.schema, send_to_embedder.clone()) {
            error!("Error transforming batches: {}", e);
        }
        drop(send_to_embedder);
        // start embedding thread
        let coordinator = EmbeddingCoordinator::new(
            num_workers,
            receive_from_embedder,
            send_to_writer,
            embedding_chunk_size,
        );
        coordinator.start();
        if let Err(e) =
            lance_writer::start_writing_thread(&store, receive_from_writer, write_buffer_size, rt)
        {
            error!("Error starting writer thread: {}", e);
        }

        Ok(())
    }
}

/// read the parquet file and send the records to the embedder
fn transform_batches(
    batches: &[RecordBatch],
    schema: &Schema,
    send_to_embedder: Sender<Vec<String>>,
) -> anyhow::Result<()> {
    // Process each batch
    for batch in batches {
        let mut records = Vec::new();
        // Process each record
        for record_idx in 0..batch.num_rows() {
            let mut record_fields = Vec::new();

            // Process each column using the helper function
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let value = extract_value_from_array(batch.column(col_idx).as_ref(), record_idx);
                record_fields.push(format!("{} is {}", field.name(), value));
            }
            let record = record_fields.join("; ");
            records.push(record);
        }
        if let Err(e) = send_to_embedder.send(records) {
            error!("Error sending batch to embedder: {}", e);
        }
    }
    // this will be dropped anyways due to scope but adding this to be explicit
    drop(send_to_embedder);
    Ok(())
}

// Helper function to extract a string representation of a value from an Arrow array for a given row
fn extract_value_from_array(array: &dyn arrow::array::Array, row_idx: usize) -> String {
    match array.data_type() {
        arrow::datatypes::DataType::Utf8 => {
            let string_array = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            string_array.value(row_idx).to_string()
        }
        arrow::datatypes::DataType::LargeUtf8 => {
            let string_array = array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap();
            string_array.value(row_idx).to_string()
        }
        arrow::datatypes::DataType::Int32 => {
            let int_array = array
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            int_array.value(row_idx).to_string()
        }
        arrow::datatypes::DataType::Int64 => {
            let int_array = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();
            int_array.value(row_idx).to_string()
        }
        arrow::datatypes::DataType::Float64 => {
            let float_array = array
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            float_array.value(row_idx).to_string()
        }
        arrow::datatypes::DataType::Boolean => {
            let bool_array = array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            bool_array.value(row_idx).to_string()
        }
        // Add more type handlers as needed
        dt => format!("[unhandled type: {}]", dt),
    }
}
