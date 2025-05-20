import subprocess
import polars as pl
import pyarrow as pa
import shutil
import os
import pathlib
import pytest
# run with `pytest -s -v tests/test_integration.py`
# Assuming DfEmbedder is importable from the 'dfembed' package/module
# Adjust the import below if your structure is different
# from dfembed import DfEmbedder # Moved inside the test function

# Define paths relative to the test file location
# Assumes the test file is in tests/ and the project root is the parent
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
# Path to the specific wheel file, relative to the project root
SPECIFIC_WHEEL_PATH = PROJECT_ROOT / "target" / "wheels" / "dfembed-0.1.4-cp310-cp310-macosx_11_0_arm64.whl"
# Assumes test data is in test-data/
TEST_DATA_DIR = PROJECT_ROOT / "test-data"
TMDB_CSV = TEST_DATA_DIR / "tmdb.csv"
# Use a distinct name for the test database
DB_NAME = "test_integration_db"
DB_DIR = PROJECT_ROOT / TEST_DATA_DIR /  DB_NAME
TABLE_NAME = "tmdb_table"
TARGET_ID = 278927
NUM_ROWS_TO_SCAN = 100 # Number of initial rows to include along with the target ID row

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """
    Fixture to install the package from a specific wheel file and manage the test database directory.
    Runs once per module.
    """
    # --- Setup ---
    # Use the specific wheel path defined above
    wheel_path = SPECIFIC_WHEEL_PATH

    # Ensure the specific wheel file exists
    if not wheel_path.exists():
        pytest.fail(f"Specified wheel file not found: {wheel_path}. Please ensure the file exists at this location.")
        return

    # Install the package using pip
    print(f"\nInstalling package from specific wheel: {wheel_path}")
    install_command = ["pip", "install", "--force-reinstall", str(wheel_path)]
    # Using check=True raises CalledProcessError on failure
    try:
        result = subprocess.run(install_command, capture_output=True, text=True, check=True, encoding='utf-8')
        print("PIP Install STDOUT:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("PIP Install STDOUT:")
        print(e.stdout)
        print("PIP Install STDERR:")
        print(e.stderr)
        pytest.fail(f"Failed to install package from {wheel_path}. pip exit code: {e.returncode}")
        return
    except FileNotFoundError:
         pytest.fail("`pip` command not found. Make sure pip is installed and in your PATH.")
         return


    # Clean up any previous database directory before tests run
    if DB_DIR.exists():
        print(f"Removing existing database directory: {DB_DIR}")
        shutil.rmtree(DB_DIR)

    # Ensure the test data directory exists (create if not)
    # Although the test checks for the file later, ensure the dir exists for clarity
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    yield # This is where the tests run

    # --- Teardown ---
    # Clean up the database directory after tests run
    if DB_DIR.exists():
        print(f"Removing database directory after test: {DB_DIR}")
        shutil.rmtree(DB_DIR)
    else:
        print(f"Database directory {DB_DIR} not found during teardown (may indicate a test failure).")


def test_tmdb_integration():
    """
    Runs an integration test for DfEmbedder:
    1. Loads TMDB data (first N rows + specific ID).
    2. Initializes DfEmbedder.
    3. Indexes the data into a table.
    4. Performs a similarity search.
    5. Asserts that a specific ID is found in the results.
    """
    # Import here, after the fixture has run and installed the package
    from dfembed import DfEmbedder

    # Check if the required data file exists
    if not TMDB_CSV.exists():
         pytest.fail(f"Test data file not found: {TMDB_CSV}. Please place it in {TEST_DATA_DIR}.")
         return

    # 1. Load and prepare data
    print(f"Loading data from: {TMDB_CSV}")
    try:
        # Use Polars to scan the CSV and filter
        # Filter logic: include the first `NUM_ROWS_TO_SCAN` rows OR the row matching TARGET_ID
        df = pl.scan_csv(TMDB_CSV).filter(
            (pl.int_range(0, pl.len()).over(None) < NUM_ROWS_TO_SCAN) | (pl.col("id") == TARGET_ID)
        ).collect()
    except Exception as e:
        # Catch potential Polars errors (e.g., file not found, column missing)
        pytest.fail(f"Failed to load or filter CSV {TMDB_CSV}: {e}")
        return

    # Verify the dataframe is not empty and contains the target ID
    if df.height == 0:
         pytest.fail(f"DataFrame is empty after loading and filtering {TMDB_CSV}. Check filters and data content.")
         return
    if df.filter(pl.col("id") == TARGET_ID).height == 0:
         pytest.fail(f"Target ID {TARGET_ID} not found in the filtered data from {TMDB_CSV}. Check the CSV and filter logic.")
         return

    print(f"Loaded dataframe shape: {df.shape}")
    try:
        arrow_table = df.to_arrow()
        print(f"Converted DataFrame to Arrow table successfully.")
    except Exception as e:
        pytest.fail(f"Failed to convert DataFrame to Arrow table: {e}")
        return

    # 2. Initialize DfEmbedder
    # The setup fixture ensures DB_DIR is clean before this point
    print(f"Initializing DfEmbedder with database directory: {DB_DIR}")
    try:
        # Pass the database *directory* path as a string
        embedder = DfEmbedder(database_name=str(DB_DIR))
    except Exception as e:
        pytest.fail(f"Failed to initialize DfEmbedder with database '{DB_DIR}': {e}")
        return

    # 3. Index the table
    print(f"Indexing data into table: {TABLE_NAME}")
    try:
        embedder.index_table(arrow_table, table_name=TABLE_NAME)
        print(f"Table '{TABLE_NAME}' indexed successfully.")
    except Exception as e:
        pytest.fail(f"Failed to index table '{TABLE_NAME}' in database '{DB_DIR}': {e}")
        return

    # Verify that the database directory and table subdirectory were created
    assert DB_DIR.exists(), f"Database directory {DB_DIR} was not created after indexing."
    # Check for the directory with the .lance extension
    expected_table_path = DB_DIR / (TABLE_NAME + ".lance")
    assert expected_table_path.exists(), f"Table directory {expected_table_path} was not created after indexing."
    print(f"Verified database and table directories exist.")

    # 4. Perform similarity search
    query = "adventures jungle animals"
    num_results = 3
    print(f"Performing similarity search for '{query}' in table '{TABLE_NAME}' (top {num_results})")
    try:
        results = embedder.find_similar(query, TABLE_NAME, num_results)
        print(f"Search results: {results}")
    except Exception as e:
        pytest.fail(f"Similarity search failed for query '{query}' in table '{TABLE_NAME}': {e}")
        return

    # 5. Assert results
    # Check if the result is a list and that the target ID is in one of the result strings
    assert isinstance(results, list), f"Expected results to be a list, but got {type(results)}: {results}"
    found = False
    for result_string in results:
        if isinstance(result_string, str) and str(TARGET_ID) in result_string:
            found = True
            break
    assert found, \
        f"Assertion Failed: Expected target ID {TARGET_ID} to be present in one of the strings in the results list.\nResults:\n{results}"
    print(f"Assertion passed: Target ID {TARGET_ID} found in search results.")

    # 6. Embed a string
    text = "adventures jungle animals"
    embedding = embedder.embed_string(text)
    assert isinstance(embedding, list), f"Expected embedding to be a list, but got {type(embedding)}: {embedding}"
    assert len(embedding) == 1024, f"Expected embedding to be of length 1024, but got {len(embedding)}: {embedding}"
    

# Optional: A separate small test to specifically check directory creation,
# though the main test already includes assertions for this.
def test_db_directory_created_after_indexing():
     """Checks if the database and table directories exist after indexing."""
     # This test relies on the main test having run successfully via the module fixture
     if not DB_DIR.exists():
          pytest.skip(f"Skipping DB directory check as {DB_DIR} does not exist (likely prior test failure).")
     assert DB_DIR.exists(), f"Database directory {DB_DIR} should exist after indexing."
     # Check for the directory with the .lance extension
     expected_table_path = DB_DIR / (TABLE_NAME + ".lance")
     assert expected_table_path.exists(), f"Table directory {expected_table_path} should exist after indexing."
     print("DB directory creation test passed.") 