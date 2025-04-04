use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use crossbeam::channel;
use crossbeam::channel::Receiver;
use crossbeam::channel::Sender;
use lance::dataset::fragment::write;
use rayon::ThreadPool;
use tokio::runtime::Runtime;
use tracing::error;
use tracing::info;

use crate::embedding::static_embeder::Embedder;
use crate::storage::lance::LanceStore;

#[derive(Debug)]
pub struct EmbeddingBatch {
    pub texts: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
}

pub struct Indexer {
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
}

const WRITER_BATCH_SIZE: usize = 2000;

impl Indexer {
    pub fn new(batches: &[RecordBatch], schema: Arc<Schema>) -> Self {
        let _ = Embedder::new().unwrap();
        Self {
            batches: batches.to_vec(),
            schema,
        }
    }
    /// Runs the indexing pipeline, processing batches of text data through embedding and storage.
    ///
    /// This function orchestrates the following workflow:
    /// 1. Transforms Arrow record batches into text chunks
    /// 2. Spawns multiple embedding worker threads that:
    ///    - Receive text chunks from a channel
    ///    - Generate embeddings using the embedding model
    ///    - Send results to a writer channel
    /// 3. Runs a writer thread that stores the embeddings and metadata in a Lance database
    ///
    /// The pipeline uses channels for communication between stages and a thread pool
    /// for parallel embedding processing.
    ///
    /// # Returns
    /// - `Ok(())` if indexing completes successfully
    /// - `Err` if there are any errors during processing
    ///
    /// # Errors
    /// This function can error if:
    /// - There are issues with the embedding model
    /// - The Lance storage operations fail
    /// - Channel communication breaks down
    pub fn run(&self, num_workers: usize, embedding_chunk_size: usize) -> anyhow::Result<()> {
        // Initialize Tokio runtime for the writer thread
        info!(
            "Starting indexer with {} workers and embedding chunk size {}",
            num_workers, embedding_chunk_size
        );
        let rt = Runtime::new()?;
        let threadpool = rayon::ThreadPoolBuilder::new().build().unwrap();
        let (send_to_embedder, receive_from_embedder) = channel::unbounded();
        let (send_to_writer, receive_from_writer) = channel::unbounded();
        let store = LanceStore::new("test.lance", 1024);

        if let Err(e) = transform_batches(&self.batches, &self.schema, send_to_embedder.clone()) {
            error!("Error transforming batches: {}", e);
        }
        drop(send_to_embedder);
        // start embedding thread
        for _ in 0..num_workers {
            let receive_from_embedder = receive_from_embedder.clone();
            let send_to_writer_clone = send_to_writer.clone();

            threadpool.spawn(move || {
                let thread_id = std::thread::current().id();
                info!("Starting embedding thread id {:?}", thread_id);
                let embed_model_clone = Embedder::new().unwrap();
                info!("Created embedder for thread id {:?}", thread_id);
                process_records(
                    receive_from_embedder,
                    send_to_writer_clone,
                    embedding_chunk_size,
                    &embed_model_clone,
                );
                info!(
                    "Embedding thread id {:?} finished .. closing channel",
                    thread_id
                );
            });
        }
        // Drop the original sender after spawning all workers
        drop(send_to_writer);

        // start the writing thread on the main thread
        while let Ok(embedding_batch) = receive_from_writer.recv() {
            let texts: Vec<&str> = embedding_batch.texts.iter().map(|s| s.as_str()).collect();
            // Block on the async add_vectors call
            if let Err(e) =
                rt.block_on(store.add_vectors(&texts, &texts, embedding_batch.embeddings))
            {
                error!("Error adding vectors: {}", e);
            }
        }
        info!("Writer thread finished - closing channel");
        drop(receive_from_writer);

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
    Ok(())
}

/// this method will continously receive records from the embedder, embed and then send the embeddings to the writer
fn process_records(
    receive_from_embedder: Receiver<Vec<String>>,
    send_to_writer: Sender<EmbeddingBatch>,
    embedding_chunk_size: usize,
    model: &Embedder,
) {
    while let Ok(records) = receive_from_embedder.recv() {
        records
            .chunks(embedding_chunk_size)
            .for_each(|chunk| match embed_chunk(chunk, model) {
                Err(e) => {
                    error!("Error embedding chunk: {}", e);
                }
                Ok(embeddings) => {
                    let embedding_batch = EmbeddingBatch {
                        texts: chunk.to_vec(),
                        embeddings,
                    };
                    if let Err(e) = send_to_writer.send(embedding_batch) {
                        error!("Error sending batch to writer: {}", e);
                    }
                }
            });
    }
    info!("Embedding thread finished.. closing channel");
    drop(send_to_writer);
}

/// process the lines in batches and return the embeddings
fn embed_chunk(chunk: &[String], model: &Embedder) -> anyhow::Result<Vec<Vec<f32>>> {
    let chunk_as_str: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
    let embeddings = model.embed_batch(&chunk_as_str).unwrap();
    // convert this to a vec<vec<f32>>
    let embeddings_vec: Vec<Vec<f32>> = embeddings
        .outer_iter() // Iterate over rows
        .map(|row| row.to_vec()) // Convert each row to Vec<f32>
        .collect(); //count the embeddings

    Ok(embeddings_vec)
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
