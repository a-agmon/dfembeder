use crate::storage::lance::LanceStore;
use crossbeam::channel::Receiver;
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use tracing::error;
use tracing::info;

#[derive(Debug)]
pub struct EmbeddingBatch {
    pub texts: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
}

pub fn start_writing_thread(
    store: &LanceStore,
    receive_from_writer: Receiver<EmbeddingBatch>,
    write_buffer_size: usize,
    rt: Arc<tokio::runtime::Runtime>,
) -> anyhow::Result<()> {
    info!("Starting writer thread id {:?}", thread::current().id());
    let mut write_buffer = EmbeddingBatch {
        texts: Vec::new(),
        embeddings: Vec::new(),
    };

    while let Ok(embedding_batch) = receive_from_writer.recv() {
        write_buffer.texts.extend(embedding_batch.texts);
        write_buffer.embeddings.extend(embedding_batch.embeddings);

        if write_buffer.texts.len() >= write_buffer_size {
            if let Err(e) = write_embedding_buffer(store, &mut write_buffer, &rt) {
                error!("Error writing embedding buffer: {}", e);
            }
        }
    }
    if write_buffer.texts.len() > 0 {
        if let Err(e) = write_embedding_buffer(store, &mut write_buffer, &rt) {
            error!("Error writing remaining embedding buffer: {}", e);
        }
    }
    info!("Writer thread finished - closing channel");
    drop(receive_from_writer);
    Ok(())
}

fn write_embedding_buffer(
    store: &LanceStore,
    embedding_buffer: &mut EmbeddingBatch,
    rt: &Arc<tokio::runtime::Runtime>,
) -> anyhow::Result<()> {
    let texts: Vec<&str> = embedding_buffer.texts.iter().map(|s| s.as_str()).collect();
    rt.block_on(store.add_vectors(&texts, &texts, embedding_buffer.embeddings.clone()))?;
    embedding_buffer.texts.clear();
    embedding_buffer.embeddings.clear();
    Ok(())
}

pub fn start_parallel_writers_shared(
    store: Arc<LanceStore>,
    receiver: Receiver<EmbeddingBatch>,
    write_buffer_size: usize,
    num_writers: usize,
    rt: Arc<tokio::runtime::Runtime>,
) -> Vec<JoinHandle<anyhow::Result<()>>> {
    (0..num_writers)
        .map(|_| {
            let store = Arc::clone(&store);
            let receiver = receiver.clone();
            let rt = Arc::clone(&rt);
            std::thread::spawn(move || {
                start_writing_thread(&store, receiver, write_buffer_size, rt)
            })
        })
        .collect()
}
