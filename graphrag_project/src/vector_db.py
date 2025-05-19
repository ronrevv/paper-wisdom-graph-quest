
"""
Vector database module for GraphRAG.
This module handles storing and retrieving vector embeddings using FAISS.
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Union

class VectorDB:
    def __init__(self, index_dir: str = "data/indices"):
        """
        Initialize the vector database.
        
        Args:
            index_dir (str): Directory to store indices
        """
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index = None
        self.chunk_ids = []
        self.chunks = []
    
    def create_index(self, embeddings: np.ndarray, chunks: List[str], chunk_ids: List[Union[int, str]] = None) -> None:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings (np.ndarray): Matrix of embeddings
            chunks (List[str]): List of text chunks corresponding to the embeddings
            chunk_ids (List[Union[int, str]], optional): List of chunk IDs. If None, will use indices.
        """
        # Get embedding dimension
        d = embeddings.shape[1]
        
        # Create FAISS index - using L2 distance by default
        self.index = faiss.IndexFlatL2(d)
        
        # Add vectors to the index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and their IDs
        self.chunks = chunks
        self.chunk_ids = chunk_ids if chunk_ids is not None else list(range(len(chunks)))
        
        print(f"Index created with {len(chunks)} chunks and {d} dimensions")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[int], List[float], List[str]]:
        """
        Search the index for the k nearest neighbors of the query vector.
        
        Args:
            query_vector (np.ndarray): The query embedding
            k (int): Number of nearest neighbors to retrieve
            
        Returns:
            Tuple containing:
                - List[int]: Indices of the nearest chunks
                - List[float]: Distances to the nearest chunks
                - List[str]: The retrieved chunks
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        # Make sure we're not requesting more neighbors than there are vectors
        k = min(k, len(self.chunks))
        
        # Search the index
        distances, indices = self.index.search(query_vector.astype('float32').reshape(1, -1), k)
        
        # Get the corresponding chunks
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        
        # Get the corresponding chunk IDs
        retrieved_ids = [self.chunk_ids[idx] for idx in indices[0]]
        
        return retrieved_ids, distances[0].tolist(), retrieved_chunks
    
    def save_index(self, index_name: str) -> None:
        """
        Save the FAISS index and associated data to disk.
        
        Args:
            index_name (str): Name for the saved index
        """
        if self.index is None:
            raise ValueError("No index to save. Call create_index first.")
        
        # Create directory path if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        metadata_path = os.path.join(self.index_dir, f"{index_name}.meta")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata (chunks and chunk_ids)
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_ids': self.chunk_ids
            }, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_name: str) -> None:
        """
        Load a FAISS index and associated data from disk.
        
        Args:
            index_name (str): Name of the saved index
        """
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        metadata_path = os.path.join(self.index_dir, f"{index_name}.meta")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index files not found for {index_name}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.chunks = metadata['chunks']
            self.chunk_ids = metadata['chunk_ids']
        
        print(f"Index loaded from {index_path} with {len(self.chunks)} chunks")
