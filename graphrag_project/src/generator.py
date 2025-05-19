
"""
Answer generation module for GraphRAG.
This module handles generating answers from retrieved chunks using OpenAI's API.
"""

import os
from typing import List, Dict, Any, Union
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Generator:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the answer generator.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will look for OPENAI_API_KEY env variable.
            model (str): The OpenAI model to use
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided either directly or via OPENAI_API_KEY environment variable")
        
        # Set up OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_answer(self, query: str, context_chunks: List[str], max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate an answer based on the query and context chunks.
        
        Args:
            query (str): The user's question
            context_chunks (List[str]): Retrieved text chunks to use as context
            max_tokens (int): Maximum number of tokens in the generated answer
            
        Returns:
            Dict with:
                - 'answer': The generated answer
                - 'model': The model used
                - 'usage': Token usage statistics
        """
        # Combine chunks into a single context string
        combined_context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. Answer the question based ONLY on the provided context. Be concise but thorough. If the context doesn't contain the answer, say 'I don't have enough information to answer this question based on the provided context.'"},
            {"role": "user", "content": f"Context information is below:\n\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Extract usage statistics
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return {
                'answer': answer,
                'model': self.model,
                'usage': usage
            }
            
        except Exception as e:
            # Handle API errors
            error_msg = f"Error generating answer: {str(e)}"
            print(error_msg)
            
            return {
                'answer': "Sorry, I couldn't generate an answer due to an error.",
                'model': self.model,
                'error': error_msg
            }
