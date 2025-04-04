use arrow::array::{FixedSizeListArray, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures::StreamExt;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub struct LanceStore {
    schema: Arc<Schema>,
    file_path: String,
    vec_dim: usize,
}

impl LanceStore {
    pub fn new(file_path: &str, vector_dim: usize) -> Self {
        Self {
            file_path: file_path.to_string(),
            vec_dim: vector_dim,
            schema: Self::get_default_schema(vector_dim),
        }
    }

    pub async fn add_vectors(
        &self,
        file_name: &[&str],
        text: &[&str],
        vectors: Vec<Vec<f32>>,
    ) -> anyhow::Result<()> {
        let key_array = StringArray::from_iter_values(file_name);
        let text_array = StringArray::from_iter_values(text);
        let vectors_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vectors
                .into_iter()
                .map(|v| Some(v.into_iter().map(|i| Some(i)))),
            self.vec_dim as i32,
        );
        let batches = vec![
            Ok(RecordBatch::try_new(
                self.schema.clone(),
                vec![
                    Arc::new(key_array),
                    Arc::new(text_array),
                    Arc::new(vectors_array),
                ],
            )?)
            .map_err(|e: Box<dyn std::error::Error + Send + Sync>| {
                ArrowError::from_external_error(e)
            }),
        ];
        let batch_iterator = RecordBatchIterator::new(batches, self.schema.clone());
        // Define write parameters (e.g. overwrite dataset)
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        Dataset::write(batch_iterator, &self.file_path, Some(write_params))
            .await
            .unwrap();
        Ok(())
    }

    /// Get the default schema for the VecDB
    pub fn get_default_schema(vector_dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("filename", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vector_dim as i32,
                ),
                true,
            ),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_vectors() {
        // Create a temporary test file path
        let test_file = "test_vectors.lance";

        // Clean up any existing test file
        if Path::new(test_file).exists() {
            fs::remove_dir_all(test_file).unwrap();
        }

        // Define test data
        let filenames = ["doc1.txt", "doc2.txt", "doc3.txt"];
        let texts = [
            "This is document 1",
            "This is document 2",
            "This is document 3",
        ];
        let vector_dim = 3;
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        // Initialize LanceStore
        let store = LanceStore::new(test_file, vector_dim);

        // Add vectors
        let result = store.add_vectors(&filenames, &texts, vectors).await;

        // Verify the operation succeeded
        assert!(result.is_ok(), "Failed to add vectors: {:?}", result.err());

        // Verify the file was created
        assert!(Path::new(test_file).exists(), "Lance file was not created");

        // Clean up the test file
        fs::remove_dir_all(test_file).unwrap();

        // Verify the file was deleted
        assert!(
            !Path::new(test_file).exists(),
            "Failed to clean up test file"
        );
    }
}
