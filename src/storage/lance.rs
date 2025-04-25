use arrow::array::{FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;

use crate::embedding::static_embeder::Embedder;



pub struct LanceStore {
    schema: Arc<Schema>,
    file_path: String,
    vec_dim: usize,
}

const VECTOR_COLUMN: &str = "vector";
const TEXT_COLUMN: &str = "text";

impl LanceStore {
    /// Creates a new LanceStore instance within a specified database directory.
    ///
    /// This constructor is designed for scenarios where Lance datasets are organized
    /// within a directory structure representing a database.
    ///
    /// # Arguments
    ///
    /// * `database_name` - The name or path of the directory to store the Lance table.
    ///                     If it doesn't contain path separators ('/' or '\'), it's treated as a directory name.
    ///                     Otherwise, it's treated as a full path.
    /// * `table_name` - The name of the Lance table (without the `.lance` extension).
    /// * `vector_dim` - The dimensionality of the vectors to be stored.
    ///
    /// # Returns
    ///
    /// A `LanceStore` instance configured with the specified schema and file path.
    pub fn new_with_database(database_name: &str, table_name: &str, vector_dim: usize) -> Self {
        let table_name = format!("{}.lance", table_name);
        let mut path_buf = PathBuf::from(database_name);
        path_buf.push(table_name);

        let file_path = path_buf
            .to_str()
            .expect("Failed to convert path to string")
            .to_string();

        Self {
            schema: Self::get_default_schema(vector_dim),
            file_path: file_path,
            vec_dim: vector_dim,
        }
    }

    pub async fn find_most_similar(
        &self,
        query: &str,
        k: usize,
        embedder: &Embedder,
    ) -> anyhow::Result<Vec<String>> {
        let query_embedding = embedder.embed_batch_vec(&[query])?;
        let query_embedding = match query_embedding.first() {
            Some(embedding) => embedding,
            None => return Err(anyhow::anyhow!("Embedder returned no vector for the query")),
        };
        let db = Dataset::open(&self.file_path).await?;

        let query_embedding_arrow = Float32Array::from(query_embedding.clone());

        // Configure the scanner first
        let mut scanner = db.scan();
        scanner.project(&[TEXT_COLUMN])?;
        scanner.nearest(VECTOR_COLUMN, &query_embedding_arrow, k)?;

        // Convert scanner to stream and collect
        let results_batches = scanner
            .try_into_stream()
            .await? // Get the stream
            .try_collect::<Vec<_>>() // Collect RecordBatches from the stream
            .await?;

        debug!("Found {} similar results.", results_batches.len());
        let mut found_texts: Vec<String> = Vec::new();

        for batch in results_batches {
            // Iterate directly over collected Vec<RecordBatch>
            // info!("Batch: {:?}", batch);
            // Example: Access the 'text' column (assuming it's the first projected column)
            if let Some(text_col) = batch.column(0).as_any().downcast_ref::<StringArray>() {
                for text_val_opt in text_col.iter() {
                    if let Some(text_val) = text_val_opt {
                        debug!("  Found similar text: {}", text_val);
                        found_texts.push(text_val.to_string());
                    }
                }
            }
        }

        Ok(found_texts)
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
                "vector",
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
    use crate::embedding::static_embeder::Embedder;
    use std::fs;
    use std::path::Path;

    #[tokio::test]
    async fn test_add_vectors() {
        // Define test database and table names
        let test_db = "test_db_add_vectors";
        let test_table_name = "test_vectors";
        let test_lance_file = format!("{}.lance", test_table_name);
        let expected_path = Path::new(test_db).join(&test_lance_file);

        // Clean up any existing test directory
        if Path::new(test_db).exists() {
            fs::remove_dir_all(test_db).unwrap();
        }
        // Create the database directory for the test
        fs::create_dir_all(test_db).expect("Failed to create test database directory");

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

        // Initialize LanceStore using the test db and table names
        let store = LanceStore::new_with_database(test_db, test_table_name, vector_dim);

        // Add vectors
        let result = store.add_vectors(&filenames, &texts, vectors).await;

        // Verify the operation succeeded
        assert!(result.is_ok(), "Failed to add vectors: {:?}", result.err());

        // Verify the lance directory/file was created at the expected path
        assert!(
            expected_path.exists(),
            "Lance dataset directory was not created at {:?}",
            expected_path
        );

        // Clean up the test directory
        fs::remove_dir_all(test_db).unwrap();

        // Verify the directory was deleted
        assert!(
            !Path::new(test_db).exists(),
            "Failed to clean up test directory"
        );
    }

    #[test]
    fn test_new_with_database_paths() {
        let table_name = "test_table";
        let vector_dim = 3;
        let lance_file_name = format!("{}.lance", table_name);

        // 1. Test with a simple directory name
        let simple_db_name = "my_db";
        let store1 = LanceStore::new_with_database(simple_db_name, table_name, vector_dim);
        let mut expected_path1_buf = PathBuf::from(simple_db_name);
        expected_path1_buf.push(&lance_file_name);
        assert_eq!(store1.file_path, expected_path1_buf.to_str().unwrap());

        // 2. Test with a path using OS-specific separators
        let temp_dir_os = tempfile::Builder::new()
            .prefix("test_os")
            .tempdir()
            .unwrap();
        let os_path_str = temp_dir_os.path().to_str().unwrap();
        let store2 = LanceStore::new_with_database(os_path_str, table_name, vector_dim);
        let mut expected_path2_buf = PathBuf::from(os_path_str);
        expected_path2_buf.push(&lance_file_name);
        assert_eq!(
            store2.file_path,
            expected_path2_buf.to_str().unwrap(),
            "Test failed for OS-specific path"
        );

        // We don't need separate tests for forward/back slashes anymore,
        // as PathBuf handles the OS-specific logic.

        // Temp directory is automatically cleaned up when `temp_dir_os` goes out of scope
    }

    #[tokio::test]
    async fn test_find_most_similar() {
        // Define test database and table names
        let test_db = "test_db";
        let test_table_name = "test_table";
        let test_lance_file = format!("{}.lance", test_table_name);
        let _expected_path = Path::new(test_db).join(&test_lance_file);

        // Clean up any existing test directory
        if Path::new(test_db).exists() {
            fs::remove_dir_all(test_db).unwrap();
        }
        // Create the database directory for the test
        fs::create_dir_all(test_db).expect("Failed to create test database directory");

        // Initialize the real Embedder
        let embedder = Embedder::new().expect("Failed to create real Embedder for test");
        let vector_dim = 1024;

        // Define test data
        let filenames = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt"];
        let texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Exploring the vast universe and its mysteries.",
            "A delicious recipe for homemade apple pie.",
            "Cats and dogs are popular domestic animals.",
        ];

        // Generate embeddings using the real Embedder
        let vectors = embedder
            .embed_batch_vec(&texts)
            .expect("Failed to generate embeddings for test data");
        assert_eq!(
            vectors.len(),
            texts.len(),
            "Number of vectors should match number of texts"
        );
        assert!(
            vectors.iter().all(|v| v.len() == vector_dim),
            "All vector dimensions should match embedder dimension"
        );

        // Initialize LanceStore with the correct dimension
        let store = LanceStore::new_with_database(test_db, test_table_name, vector_dim);

        // Add vectors generated by the real embedder
        store
            .add_vectors(&filenames, &texts, vectors)
            .await
            .expect("Failed to add real vectors for similarity test");

        // Define the query text - semantically similar to the 4th text
        let query_text = "Information about pets like felines and canines";
        let k = 1;

        // Use the actual find_most_similar function
        let found_texts_result = store.find_most_similar(query_text, k, &embedder).await;

        // Verify the results
        assert!(
            found_texts_result.is_ok(),
            "find_most_similar failed: {:?}",
            found_texts_result.err()
        );
        let found_texts = found_texts_result.unwrap();

        println!("Query: '{}'", query_text);
        println!("Found texts: {:?}", found_texts);

        assert_eq!(
            found_texts.len(),
            k,
            "Expected {} result(s), found {}",
            k,
            found_texts.len()
        );
        assert!(
            found_texts.contains(&texts[3].to_string()),
            "Expected '{}' to be the most similar, but found {:?}",
            texts[3],
            found_texts
        );

        // Clean up the test directory
        fs::remove_dir_all(test_db).unwrap();
        assert!(
            !Path::new(test_db).exists(),
            "Failed to clean up test directory"
        );
    }
}
