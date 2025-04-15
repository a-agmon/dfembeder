"""
dfembed - A Python library using Rust to index Arrow tables.
"""
import importlib  # Needed for __getattr__

# Import the core class directly, this should always be available
from .core import DfEmbedder

__version__ = "0.1.0"

# Define what 'from dfembed import *' imports (keeping DfEmbedVectorStore for discoverability)
__all__ = [
    # Core Python wrapper class
    "DfEmbedder",
    # LlamaIndex Vector Store (Requires llama_index to be installed)
    "DfEmbedVectorStore",
]

# --- Lazy Loading for Optional Dependencies ---

def __getattr__(name: str):
    """
    Lazily load optional components like the LlamaIndex vector store
    only when they are requested.
    """
    if name == "DfEmbedVectorStore":
        try:
            # Attempt to import the submodule and class
            llamaindex_module = importlib.import_module(".llamaindex", __name__)
            # Get the class from the loaded module
            DfEmbedVectorStore = getattr(llamaindex_module, "DfEmbedVectorStore")
            # Store it in the module's globals dict so this runs only once
            globals()[name] = DfEmbedVectorStore
            return DfEmbedVectorStore
        except ImportError as e:
            # Reraise with a helpful message if llama_index is likely missing
            raise ImportError(
                f"Could not import 'DfEmbedVectorStore'. "
                f"Please ensure 'llama-index-core' is installed to use this feature. "
                f"Original error: {e}"
            ) from e
        except AttributeError:
            # Handle case where .llamaindex exists but DfEmbedVectorStore isn't in it
             raise AttributeError(f"Module '{__name__}.llamaindex' has no attribute '{name}'")

    # For any other undefined attribute, raise the default error
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")
