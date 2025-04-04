"""
Arrow Analyzer - A Python library using Rust to analyze Arrow tables.
"""

# Import functions from the underlying Rust extension module
from .arrow_analyzer import analyze_arrow_table, index_arrow_table

__version__ = "0.1.0"
# Define what 'from arrow_analyzer import *' imports
__all__ = ["analyze_arrow_table", "index_arrow_table"]