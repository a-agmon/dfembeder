use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use tracing::debug;

use std::sync::Arc;

use crossbeam::channel;
use crossbeam::channel::Receiver;
use crossbeam::channel::Sender;
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
        let rt = Runtime::new()?;
        let threadpool = rayon::ThreadPoolBuilder::new().build().unwrap();
        let (send_to_embedder, receive_from_embedder) = channel::unbounded();
        let (send_to_writer, receive_from_writer) = channel::unbounded();
        let store = LanceStore::new_with_database(database_name, table_name, vector_dim);

        // transform the batches to text chunks and send them to the embedder
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
                debug!("Starting embedding thread id {:?}", thread_id);
                let embed_model_clone = Embedder::new().unwrap();
                info!("Created embedder for thread id {:?}", thread_id);
                embed_text_chunks(
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
        let mut write_buffer = EmbeddingBatch {
            texts: Vec::new(),
            embeddings: Vec::new(),
        };
        while let Ok(embedding_batch) = receive_from_writer.recv() {
            // we add the texts and embeddings to the buffer
            write_buffer.texts.extend(embedding_batch.texts);
            write_buffer.embeddings.extend(embedding_batch.embeddings);

            if write_buffer.texts.len() >= write_buffer_size {
                // Write the buffer if it's full
                if let Err(e) = write_embedding_buffer(&store, &mut write_buffer, &rt) {
                    error!("Error writing embedding buffer: {}", e);
                }
            }
        }
        // write the remaining embeddings
        if write_buffer.texts.len() > 0 {
            // Write any remaining data in the buffer
            if let Err(e) = write_embedding_buffer(&store, &mut write_buffer, &rt) {
                error!("Error writing remaining embedding buffer: {}", e);
            }
        }
        info!("Writer thread finished - closing channel");
        drop(receive_from_writer);

        Ok(())
    }
}

/// Helper function to write the contents of the embedding buffer to the Lance store.
fn write_embedding_buffer(
    store: &LanceStore,
    embedding_buffer: &mut EmbeddingBatch,
    rt: &Runtime,
) -> anyhow::Result<()> {
    let texts: Vec<&str> = embedding_buffer.texts.iter().map(|s| s.as_str()).collect();
    // Note: We clone embeddings here to satisfy LanceStore::add_vectors signature.
    // The buffer is cleared afterwards, so this clone is temporary.
    rt.block_on(store.add_vectors(&texts, &texts, embedding_buffer.embeddings.clone()))?;
    // Clear the buffer after successful write
    embedding_buffer.texts.clear();
    embedding_buffer.embeddings.clear();
    Ok(())
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

/// this method will continously receive records from the embedder, embed and then send the embeddings to the writer
fn embed_text_chunks(
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
