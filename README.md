# Arrow Analyzer

A Python library built with Rust and PyO3 that analyzes Apache Arrow tables.

## Description

Arrow Analyzer provides functionality to analyze Arrow tables, particularly those generated from Polars DataFrames. It demonstrates how to:

- Use PyO3 to create Python bindings for Rust code
- Process Apache Arrow tables in Rust
- Integrate with Polars DataFrames via Arrow

## Requirements

- Rust (1.56+)
- Python (3.8+)
- Maturin (1.0+)
- Polars
- PyArrow

## Installation

### Development Setup

1. Clone this repository
2. Create and activate a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```
   pip install maturin polars pyarrow pytest
   ```

### Building the Package

```bash
cd arrow_analyzer
maturin develop
```

This will build the Rust library and install it into your current Python environment.

## Usage

```python
import polars as pl
from arrow_analyzer import analyze_arrow_table

# Load data with Polars
df_lazy = pl.scan_csv("your_data.csv")
df_reg = df_lazy.collect()

# Convert to Arrow table
df_arr = df_reg.to_arrow()

# Analyze the Arrow table
analyze_arrow_table(df_arr)
```

## Running Tests

```bash
cd arrow_analyzer
pytest python_tests/
```

Or run the test script directly for more verbose output:

```bash
cd arrow_analyzer
python python_tests/test_analyzer.py
```

## How It Works

1. The library uses PyO3 to create Python bindings for Rust code
2. It uses pyo3-arrow to handle Arrow data between Python and Rust
3. When `analyze_arrow_table` is called, it:
   - Converts the PyArrow table to a Rust Arrow table
   - Prints information about the schema
   - Counts and displays information about the record batches

## License

MIT