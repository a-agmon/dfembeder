use anyhow::Result;
use candle::{DType, Device, Tensor};
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use thiserror::Error;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use tracing::debug;

const MODEL_URL_BASE: &str = "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/main/0_StaticEmbedding";
const MODEL_FILES: [&str; 2] = ["model.safetensors", "tokenizer.json"];

// Default embedding dimension for the model
const DEFAULT_EMBEDDING_DIM: usize = 1024;

#[derive(Error, Debug)]
pub enum EmbedderError {
    #[error("Failed to download model file: {0}")]
    DownloadError(String),

    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Failed to tokenize input: {0}")]
    TokenizationError(String),
}

pub struct Embedder {
    _model_path: PathBuf,
    embedding_weights: Tensor,
    tokenizer: Tokenizer,
    pub embedding_dim: usize,
    device: Device,
}

impl Embedder {
    /// Create a new Embedder instance
    pub fn new() -> Result<Self> {
        let model_path = Self::ensure_model_files()?;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            EmbedderError::ModelLoadError(format!("Failed to load tokenizer: {}", e))
        })?;

        // Configure tokenizer padding
        let pp = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        // Load embedding weights from safetensors
        let device = Device::Cpu;
        let weights_path = model_path.join("model.safetensors");

        // Read the safetensors file
        let data = fs::read(&weights_path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        // Get the embedding weights tensor
        let embedding_tensor = tensors.tensor("embedding.weight")?;
        let shape = embedding_tensor.shape();
        debug!("Loaded embedding weights with shape: {:?}", shape);

        // Convert to candle tensor
        let embedding_weights = Tensor::from_raw_buffer(
            embedding_tensor.data(),
            DType::F32,
            &shape.to_vec(),
            &device,
        )?;

        Ok(Self {
            _model_path: model_path,
            embedding_weights,
            tokenizer,
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            device,
        })
    }

    /// Ensure model files are downloaded and return the path to the model directory
    fn ensure_model_files() -> Result<PathBuf> {
        let model_dir = PathBuf::from("models/static-retrieval-mrl-en-v1");

        if !model_dir.exists() {
            fs::create_dir_all(&model_dir)?;
        }

        for file in MODEL_FILES.iter() {
            let file_path = model_dir.join(file);

            if !file_path.exists() {
                println!("Downloading {}", file);
                let url = format!("{}/{}", MODEL_URL_BASE, file);
                let response = reqwest::blocking::get(&url).map_err(|e| {
                    EmbedderError::DownloadError(format!("Failed to download {}: {}", file, e))
                })?;

                let content = response.bytes().map_err(|e| {
                    EmbedderError::DownloadError(format!(
                        "Failed to read response for {}: {}",
                        file, e
                    ))
                })?;

                let parent_dir = file_path.parent().unwrap();
                if !parent_dir.exists() {
                    fs::create_dir_all(parent_dir)?;
                }

                let mut file = fs::File::create(&file_path)?;
                file.write_all(&content)?;
            }
        }

        Ok(model_dir)
    }

    /// Embed a single string (prefixed to silence unused warning)
    pub fn _embed(&self, text: &str) -> Result<Array1<f32>> {
        // First get the tensor embedding
        let tensor_embeddings = self.embed_batch_tensor(&[text])?;

        // Convert to ndarray for compatibility with existing code
        let embedding = tensor_embeddings.get(0)?;
        let embedding_vec = embedding.to_vec1::<f32>()?;
        let array_embedding = Array1::from_vec(embedding_vec);

        Ok(array_embedding)
    }

    /// Embed a batch of strings and return as ndarray (for compatibility)
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Array2<f32>> {
        // Get tensor embeddings
        let tensor_embeddings = self.embed_batch_tensor(texts)?;

        // Convert to ndarray for compatibility
        // Get dimensions manually
        let batch_size = texts.len();
        let flat_data = tensor_embeddings.flatten_all()?.to_vec1::<f32>()?;

        // The embedding dimension should be flat_data.len() / batch_size
        let embedding_dim = if batch_size > 0 {
            flat_data.len() / batch_size
        } else {
            self.embedding_dim
        };

        let array_embeddings = Array2::from_shape_vec((batch_size, embedding_dim), flat_data)?;

        Ok(array_embeddings)
    }

    /// Embed a batch of strings and return as Vec<Vec<f32>>
    pub fn embed_batch_vec(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Get tensor embeddings
        let tensor_embeddings = self.embed_batch_tensor(texts)?;

        let result = tensor_embeddings.to_vec2::<f32>()?; // Direct conversion

        Ok(result)
    }

    /// Embed a batch of strings and return as Tensor
    fn embed_batch_tensor(&self, texts: &[&str]) -> Result<Tensor> {
        // Tokenize the inputs
        let tokens = self
            .tokenizer
            .encode_batch(texts.iter().map(|s| s.to_string()).collect(), true)
            .map_err(|e| {
                EmbedderError::TokenizationError(format!("Failed to tokenize input: {}", e))
            })?;

        let mut embeddings = Vec::with_capacity(texts.len());

        for encoding in &tokens {
            let input_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            // Get embeddings for each token
            let mut token_embeddings = Vec::new();
            for (i, &token_id) in input_ids.iter().enumerate() {
                if i < attention_mask.len() && attention_mask[i] == 1 {
                    // Get embedding for this token from the embedding table
                    let token_embedding = self.embedding_weights.get(token_id as usize)?;
                    token_embeddings.push(token_embedding);
                }
            }

            // Stack token embeddings
            if token_embeddings.is_empty() {
                // If no tokens, create a zero embedding
                let zero_embedding =
                    Tensor::zeros((1, self.embedding_dim), DType::F32, &self.device)?;
                embeddings.push(zero_embedding);
            } else {
                // Stack and mean pool the token embeddings
                let stacked = Tensor::stack(&token_embeddings, 0)?;
                let mean_embedding = stacked.mean(0)?;
                embeddings.push(mean_embedding.unsqueeze(0)?);
            }
        }

        // Stack all sentence embeddings
        let result = Tensor::cat(&embeddings, 0)?;

        // Normalize embeddings
        let result = self.normalize_l2(&result)?;

        Ok(result)
    }

    /// Normalize a tensor using L2 normalization
    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor> {
        let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(v.broadcast_div(&norm)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    // Helper function for dot product (cosine similarity for normalized vectors)
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        // Ensure vectors have the same length before calculating dot product
        assert_eq!(
            a.len(),
            b.len(),
            "Vectors must have the same length for dot product"
        );
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    // Check if vector is normalized
    fn is_normalized(v: &[f32]) -> bool {
        let sum_squares: f32 = v.iter().map(|x| x * x).sum();
        (sum_squares - 1.0).abs() < 0.01 // Allow small floating point variation
    }

    #[test]
    fn test_embed_batch_vec_structure() -> Result<()> {
        // 1. Instantiate Embedder
        println!("Initializing embedder for test...");
        let embedder = Embedder::new()?;
        let expected_dim = embedder.embedding_dim;
        println!("Embedder initialized. Expected dimension: {}", expected_dim);

        // 2. Define simple test input
        let texts = vec!["Test sentence one.", "Test sentence two."];
        println!("Embedding test texts...");

        // 3. Call embed_batch_vec
        let embeddings = embedder.embed_batch_vec(&texts)?;
        println!("Embeddings generated.");

        // 4. Verify basic structure
        assert_eq!(
            embeddings.len(),
            texts.len(),
            "Incorrect number of embeddings returned."
        );
        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(
                emb.len(),
                expected_dim,
                "Embedding {} has incorrect dimension.",
                i
            );
            assert!(
                emb.iter().all(|&x| x.is_finite()),
                "Embedding {} contains non-finite values.",
                i
            );
            assert!(
                is_normalized(emb),
                "Embedding {} is not properly normalized",
                i
            );
        }
        println!(
            "Structure test passed - embeddings are correctly normalized and have the expected dimensions."
        );

        Ok(())
    }

    #[test]
    fn test_embed_batch_vec_semantic_similarity() -> Result<()> {
        // 1. Instantiate Embedder
        println!("Initializing embedder for test...");
        let embedder = Embedder::new()?;
        println!("Embedder initialized.");

        // 2. Define Test Input with more strongly differentiated semantically similar/dissimilar pairs
        let texts = vec![
            "Dogs are popular pets that need to be walked regularly.", // 0: Dog sentence
            "Canines make great companions and require daily exercise.", // 1: Similar to 0
            "Quantum mechanics is a fundamental theory in physics.",   // 2: Science sentence
            "Physics uses quantum theory to describe subatomic particles.", // 3: Similar to 2
            "Mountains are large landforms that rise prominently above their surroundings.", // 4: Unrelated to others
        ];
        println!("Embedding test texts...");

        // 3. Call embed_batch_vec
        let embeddings = embedder.embed_batch_vec(&texts)?;
        println!("Embeddings generated.");

        // 4. Calculate similarity scores
        let sim_dog_canine = dot_product(&embeddings[0], &embeddings[1]);
        let sim_quantum_physics = dot_product(&embeddings[2], &embeddings[3]);
        let sim_dog_quantum = dot_product(&embeddings[0], &embeddings[2]);
        let sim_canine_physics = dot_product(&embeddings[1], &embeddings[3]);
        let sim_dog_mountain = dot_product(&embeddings[0], &embeddings[4]);
        let sim_quantum_mountain = dot_product(&embeddings[2], &embeddings[4]);

        // 5. Print all similarity scores for debugging
        println!("Similarity (dog vs canine): {:.4}", sim_dog_canine);
        println!(
            "Similarity (quantum vs physics): {:.4}",
            sim_quantum_physics
        );
        println!("Similarity (dog vs quantum): {:.4}", sim_dog_quantum);
        println!("Similarity (canine vs physics): {:.4}", sim_canine_physics);
        println!("Similarity (dog vs mountain): {:.4}", sim_dog_mountain);
        println!(
            "Similarity (quantum vs mountain): {:.4}",
            sim_quantum_mountain
        );

        // 6. Check that related concepts have higher similarity than unrelated ones
        // Instead of absolute thresholds, use relative comparisons

        // Dog related comparisons
        assert!(
            sim_dog_canine > sim_dog_quantum,
            "Expected similarity between dog/canine to be higher than dog/quantum"
        );
        assert!(
            sim_dog_canine > sim_dog_mountain,
            "Expected similarity between dog/canine to be higher than dog/mountain"
        );

        // Physics related comparisons
        assert!(
            sim_quantum_physics > sim_dog_quantum,
            "Expected similarity between quantum/physics to be higher than dog/quantum"
        );
        assert!(
            sim_quantum_physics > sim_quantum_mountain,
            "Expected similarity between quantum/physics to be higher than quantum/mountain"
        );

        // Verify by how much (ratio check - related should be at least 2x more similar than unrelated)
        let dog_ratio = sim_dog_canine.abs() / (sim_dog_quantum.abs().max(0.01));
        let physics_ratio = sim_quantum_physics.abs() / (sim_canine_physics.abs().max(0.01));

        println!("Dog ratio (similar/dissimilar): {:.2}", dog_ratio);
        println!("Physics ratio (similar/dissimilar): {:.2}", physics_ratio);

        assert!(
            dog_ratio > 2.0,
            "Expected dog/canine similarity to be at least 2x higher than dog/quantum"
        );
        assert!(
            physics_ratio > 2.0,
            "Expected quantum/physics similarity to be at least 2x higher than dog/quantum"
        );

        println!("Semantic similarity verified with relative comparisons.");

        Ok(())
    }
}
