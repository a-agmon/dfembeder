"""
dfembed - A Python library using Rust to index Arrow tables.
"""

# Import the main Python wrapper class from the core module
from .core import DfEmbedder

__version__ = "0.1.0"

# Define what 'from dfembed import *' imports
__all__ = [
    # Core Python wrapper class
    "DfEmbedder",
    # We no longer expose the raw Rust functions directly
    # "index_arrow_table",
    # "analyze_arrow_table",
]

# We also no longer need to import the raw functions here
# import os
# from .dfembed import index_arrow_table, analyze_arrow_table