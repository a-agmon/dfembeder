use crate::{embedding::static_embeder::Embedder, storage::lance_writer::EmbeddingBatch};
use crossbeam::channel::{Receiver, Sender};
use rayon::ThreadPool;
use tracing::{debug, error, info};

pub struct EmbeddingCoordinator {
    thread_pool: ThreadPool,
    num_workers: usize,
    reciever_channel: Receiver<Vec<String>>,
    sender_channel: Sender<EmbeddingBatch>,
    embedding_chunk_size: usize,
}

impl EmbeddingCoordinator {
    pub fn new(
        num_workers: usize,
        reciever_channel: Receiver<Vec<String>>,
        sender_channel: Sender<EmbeddingBatch>,
        embedding_chunk_size: usize,
    ) -> Self {
        let threadpool = rayon::ThreadPoolBuilder::new().build().unwrap();
        Self {
            thread_pool: threadpool,
            num_workers,
            reciever_channel,
            sender_channel,
            embedding_chunk_size,
        }
    }

    pub fn start(self) {
        for _ in 0..self.num_workers {
            let receive_from_embedder = self.reciever_channel.clone();
            let send_to_writer_clone = self.sender_channel.clone();
            let chunk_size = self.embedding_chunk_size;

            self.thread_pool.spawn(move || {
                let thread_id = std::thread::current().id();
                debug!("Starting embedding thread id {:?}", thread_id);
                let embed_model_clone = Embedder::new().unwrap();
                info!("Created embedder for thread id {:?}", thread_id);
                embed_text_chunks(
                    receive_from_embedder,
                    send_to_writer_clone,
                    chunk_size,
                    &embed_model_clone,
                );
                info!(
                    "Embedding thread id {:?} finished .. closing channel",
                    thread_id
                );
            });
        }
        drop(self.sender_channel);
    }
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
