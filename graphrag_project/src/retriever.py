
"""
Retrieval module for GraphRAG.
This module handles retrieving relevant chunks based on a query using both vector search and graph traversal.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

class Retriever:
    def __init__(self, vector_db, graph_builder):
        """
        Initialize the retriever.
        
        Args:
            vector_db: Vector database instance
            graph_builder: Knowledge graph builder instance
        """
        self.vector_db = vector_db
        self.graph_builder = graph_builder
    
    def retrieve_chunks(self, 
                      query_vector: np.ndarray, 
                      top_k_vector: int = 5,
                      top_k_graph: int = 3,
                      max_graph_distance: int = 2) -> Dict[str, Any]:
        """
        Retrieve relevant chunks using vector search and graph enhancement.
        
        Args:
            query_vector (np.ndarray): Embedding of the query
            top_k_vector (int): Number of chunks to retrieve via vector search
            top_k_graph (int): Number of additional chunks to retrieve via graph
            max_graph_distance (int): Maximum distance in the graph to traverse
            
        Returns:
            Dict with:
                - 'vector_chunk_ids': IDs of chunks retrieved via vector search
                - 'vector_chunks': Text of chunks retrieved via vector search
                - 'vector_scores': Relevance scores for vector-retrieved chunks
                - 'graph_chunk_ids': IDs of chunks retrieved via graph
                - 'graph_chunks': Text of chunks retrieved via graph
                - 'all_chunk_ids': All chunk IDs (vector + graph)
                - 'all_chunks': All chunks (vector + graph)
        """
        # First, retrieve chunks using vector search
        vector_ids, vector_distances, vector_chunks = self.vector_db.search(query_vector, k=top_k_vector)
        
        # Convert scores - smaller distance is better, so we invert it for scoring (1 / (1 + distance))
        vector_scores = [1.0 / (1.0 + dist) for dist in vector_distances]
        
        # Then, enhance with graph-based retrieval
        graph_chunk_ids = self.graph_builder.get_related_chunks(
            seed_chunks=vector_ids,
            max_distance=max_graph_distance,
            top_k=top_k_graph
        )
        
        # Get the text for graph-retrieved chunks
        graph_chunks = []
        for chunk_id in graph_chunk_ids:
            # Find the chunk text from vector_db's stored chunks
            chunk_index = self.vector_db.chunk_ids.index(chunk_id) if chunk_id in self.vector_db.chunk_ids else -1
            if chunk_index >= 0:
                graph_chunks.append(self.vector_db.chunks[chunk_index])
            else:
                graph_chunks.append(f"[Chunk {chunk_id} not found]")
        
        # Combine vector and graph results
        all_chunk_ids = vector_ids + graph_chunk_ids
        all_chunks = vector_chunks + graph_chunks
        
        return {
            'vector_chunk_ids': vector_ids,
            'vector_chunks': vector_chunks,
            'vector_scores': vector_scores,
            'graph_chunk_ids': graph_chunk_ids,
            'graph_chunks': graph_chunks,
            'all_chunk_ids': all_chunk_ids,
            'all_chunks': all_chunks
        }
