
"""
Text embedding module for GraphRAG.
This module handles converting text chunks into vector embeddings.
"""

from typing import List, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedder with a sentence-transformers model.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Embedder initialized with model {model_name} (embedding dim: {self.embedding_dim})")
        except Exception as e:
            raise Exception(f"Error loading sentence-transformers model: {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        return self.model.encode(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            np.ndarray: Matrix of embeddings with shape (len(texts), embedding_dim)
        """
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_chunks(self, chunks_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Embed all chunks from a document.
        
        Args:
            chunks_data (Dict): Dictionary containing document info and chunks
            
        Returns:
            Dict: The input dictionary augmented with embeddings
        """
        chunks = chunks_data['chunks']
        embeddings = self.embed_batch(chunks)
        
        # Add embeddings to the dictionary
        chunks_data['embeddings'] = embeddings
        return chunks_data
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query (str): The query text
            
        Returns:
            np.ndarray: The query embedding vector
        """
        return self.embed_text(query)
