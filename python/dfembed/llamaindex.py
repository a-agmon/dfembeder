from typing import List, Any, Optional
import uuid

# --- LlamaIndex Core Imports ---
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.embeddings import MockEmbedding # To satisfy LlamaIndex setup

# --- Your Library Import ---
# Assume DfEmbedder is importable (replace with actual import path if needed)

from .core import DfEmbedder # Use relative import within the package

# --- Custom Vector Store Implementation ---

class DfEmbedVectorStore(VectorStore):
    """
    LlamaIndex VectorStore implementation using a DfEmbedder instance.

    Assumes data indexing is handled externally by DfEmbedder.index_table().
    Leverages DfEmbedder.find_similar() for querying.
    """
    stores_text: bool = True # We can store text in the nodes we return
    is_embedding_query: bool = False # DfEmbedder handles embedding internally

    def __init__(
        self,
        df_embedder: DfEmbedder,
        table_name: str,
        **kwargs: Any,
    ) -> None:
        """
        Initialize DfEmbedVectorStore.

        Args:
            df_embedder: An initialized instance of DfEmbedder.
            table_name: The LanceDB table name used by the embedder.
        """
        super().__init__(**kwargs)
        if not isinstance(df_embedder, DfEmbedder):
            raise TypeError("df_embedder must be an instance of DfEmbedder")
        if not table_name:
            raise ValueError("table_name must be provided")

        self._embedder = df_embedder
        self._table_name = table_name
        print(f"DfEmbedVectorStore initialized for table: '{self._table_name}'")

    @property
    def client(self) -> Any:
        """Return the underlying DfEmbedder instance."""
        return self._embedder

    def add(self, nodes: List[TextNode], **add_kwargs: Any) -> List[str]:
        """
        Adding nodes via LlamaIndex is not supported.
        Use DfEmbedder.index_table() directly before initializing the store.
        """
        print("DfEmbedVectorStore: 'add' called, raising NotImplementedError.")
        raise NotImplementedError(
            "Adding nodes directly via LlamaIndex is not supported. "
            "Use DfEmbedder.index_table() externally."
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Node deletion is not currently supported by this integration."""
        print("DfEmbedVectorStore: 'delete' called, raising NotImplementedError.")
        raise NotImplementedError("Node deletion is not currently supported.")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query the vector store using the DfEmbedder's find_similar method.

        Args:
            query: The VectorStoreQuery object from LlamaIndex.

        Returns:
            VectorStoreQueryResult containing nodes found by find_similar.
        """
        if query.query_embedding is not None:
            # This shouldn't happen if is_embedding_query=False, but good practice
            print("Warning: query.query_embedding provided but will be ignored by DfEmbedVectorStore.")

        if query.query_str is None:
             raise ValueError("Query text (query.query_str) is required for DfEmbedVectorStore.")

        # --- Core Query Logic ---
        query_text = query.query_str
        k = query.similarity_top_k
        print(f"DfEmbedVectorStore: Received query: '{query_text}', k={k}")

        # --- Metadata Filtering (Optional - Adapt if DfEmbedder supports it) ---
        # This example assumes find_similar doesn't support LlamaIndex filters.
        # If your DfEmbedder *can* filter (e.g., via SQL WHERE clause in LanceDB),
        # you would parse query.filters here and pass them to find_similar.
        if query.filters:
            print(f"Warning: Metadata filters received but not implemented in this DfEmbedVectorStore example: {query.filters}")
            # Example parsing (if you were to implement it):
            # custom_filters = self._parse_llama_filters(query.filters)
            # similar_texts = self._embedder.find_similar(..., filters=custom_filters)

        # Call DfEmbedder's find_similar
        try:
            # Note: DfEmbedder.find_similar expects k to be int. LlamaIndex might provide Optional[int].
            if k is None:
                 raise ValueError("similarity_top_k must be provided in the query.")
            similar_texts: List[str] = self._embedder.find_similar(
                query=query_text,
                table_name=self._table_name,
                k=k
            )
            print(f"DfEmbedVectorStore: find_similar returned {len(similar_texts)} results.")
        except Exception as e:
            print(f"Error during embedder.find_similar: {e}")
            raise # Re-raise the exception

        # --- Result Conversion ---
        nodes: List[TextNode] = []
        similarities: List[float] = [] # find_similar doesn't return scores, so we generate dummy ones
        ids: List[str] = [] # Initialize list to store node IDs

        for i, text_content in enumerate(similar_texts):
            # Create a unique ID for the node
            node_id = f"{self._table_name}_result_{uuid.uuid4()}"
            ids.append(node_id) # Add the generated ID to the list
            # Assign a simple rank-based score (higher rank = higher score)
            # You might get actual scores if find_similar returned them
            score = 1.0 - (i / k) if k > 0 else 1.0 # Handle k=0 edge case

            node = TextNode(
                text=text_content,
                id_=node_id,
                # You could add metadata here if find_similar returned it
                # metadata={"source_table": self._table_name}
            )
            nodes.append(node)
            similarities.append(score)

        # Pass the collected IDs to the result object
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    # Helper function if you implement filter parsing
    # def _parse_llama_filters(self, standard_filters: MetadataFilters) -> Any:
    #     # Implement logic to convert LlamaIndex filters to whatever
    #     # DfEmbedder/LanceDB expects (e.g., SQL WHERE clause string)
    #     raise NotImplementedError("Filter parsing not implemented.") 