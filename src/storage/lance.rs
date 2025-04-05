use arrow::array::{FixedSizeListArray, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures::StreamExt;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use std::fs;
use std::path::{Path, PathBuf};
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
}
