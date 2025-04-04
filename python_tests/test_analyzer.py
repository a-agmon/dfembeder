"""
Test script for the arrow_analyzer library.

This script tests the analyze_arrow_table function using a Polars DataFrame
converted to an Arrow table.
"""

import os
import sys
import polars as pl
import pytest
from arrow_analyzer import analyze_arrow_table

def test_analyze_arrow_table(capfd):
    """Test the analyze_arrow_table function with a Polars DataFrame."""
    # Get the absolute path to the test.csv file
    test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(test_dir, "tests", "test.csv")
    
    # Ensure the test file exists
    assert os.path.exists(test_file), f"Test file not found: {test_file}"
    
    # Load CSV with Polars
    df_lazy = pl.scan_csv(test_file)
    df_reg = df_lazy.collect()
    
    # Convert to Arrow table
    df_arr = df_reg.to_arrow()
    
    # Call our Rust function
    analyze_arrow_table(df_arr)
    
    # Capture stdout and verify output
    captured = capfd.readouterr()
    output = captured.out
    
    # Assert that the output contains expected information
    assert "Arrow Table Schema:" in output
    assert "Number of record batches:" in output
    assert "id: int64" in output
    assert "name: large_string" in output
    assert "age: int64" in output
    assert "score: double" in output

if __name__ == "__main__":
    # If run directly, print more detailed output
    test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(test_dir, "tests", "test.csv")
    
    print(f"Testing with file: {test_file}")
    
    # Load CSV with Polars
    df_lazy = pl.scan_csv(test_file)
    df_reg = df_lazy.collect()
    
    # Print Polars DataFrame
    print("\nPolars DataFrame:")
    print(df_reg)
    
    # Convert to Arrow table
    df_arr = df_reg.to_arrow()
    print("\nArrow Table Type:", type(df_arr))
    
    # Call our Rust function
    print("\nAnalyzing Arrow Table:")
    analyze_arrow_table(df_arr)