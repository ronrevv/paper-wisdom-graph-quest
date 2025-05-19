
"""
Text extraction and chunking module for GraphRAG.
This module handles extracting text from PDFs and splitting it into manageable chunks.
"""

import os
import re
from typing import List, Dict, Union
from PyPDF2 import PdfReader
from tqdm import tqdm

class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the Chunker.
        
        Args:
            chunk_size (int): The target size of each chunk in terms of characters
            chunk_overlap (int): The overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in tqdm(reader.pages, desc="Extracting pages"):
                text += page.extract_text() + "\n"
            
            # Clean the text
            text = self._clean_text(text)
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the extracted text by removing excessive whitespace and special characters.
        
        Args:
            text (str): The text to clean
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        return text.strip()
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split the text into overlapping chunks.
        
        Args:
            text (str): The text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            chunks.append(text)
            return chunks
        
        # Find natural break points (paragraphs, sentences) and use them for chunking
        paragraphs = re.split(r'\n\n', text)
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start a new chunk, but check if the paragraph itself is too large
                if len(paragraph) > self.chunk_size:
                    # If paragraph is too large, split it by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                        else:
                            current_chunk += sentence + " "
                else:
                    current_chunk = paragraph + "\n\n"
            else:
                current_chunk += paragraph + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap
        chunks_with_overlap = []
        for i in range(len(chunks)):
            if i == 0:
                chunks_with_overlap.append(chunks[i])
            else:
                # Take the end of the previous chunk and add it to the start of this chunk
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                chunks_with_overlap.append(overlap_text + " " + chunks[i])
        
        return chunks_with_overlap
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Process a PDF file: extract text and split into chunks.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict with keys:
                'filename': The PDF filename
                'full_text': The complete extracted text
                'chunks': List of text chunks
        """
        filename = os.path.basename(pdf_path)
        full_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_into_chunks(full_text)
        
        return {
            'filename': filename,
            'full_text': full_text,
            'chunks': chunks
        }
