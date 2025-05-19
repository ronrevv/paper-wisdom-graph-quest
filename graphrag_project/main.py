
"""
Main script for GraphRAG system.
This script combines all components to create a complete Retrieval-Augmented Generation system
with knowledge graph enhancement.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import the GraphRAG components
from src.chunker import Chunker
from src.embedder import Embedder
from src.vector_db import VectorDB
from src.graph_builder import GraphBuilder
from src.retriever import Retriever
from src.generator import Generator
from src.evaluator import Evaluator

# Load environment variables
load_dotenv()

def ensure_directories_exist():
    """Create necessary directories."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/indices", exist_ok=True)
    os.makedirs("data/graphs", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

class GraphRAG:
    """Main GraphRAG system class that integrates all components."""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt-3.5-turbo",
                 spacy_model: str = "en_core_web_sm",
                 chunk_size: int = 500, 
                 chunk_overlap: int = 100):
        """
        Initialize the GraphRAG system.
        
        Args:
            embedding_model (str): Name of the sentence-transformer model to use
            llm_model (str): Name of the large language model to use for answer generation
            spacy_model (str): Name of the spaCy model to use for entity extraction
            chunk_size (int): Size of text chunks in characters
            chunk_overlap (int): Overlap between chunks in characters
        """
        # Initialize components
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_db = VectorDB()
        self.graph_builder = GraphBuilder(spacy_model=spacy_model)
        self.retriever = Retriever(self.vector_db, self.graph_builder)
        
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found. Answer generation will not work.")
            self.generator = None
        else:
            self.generator = Generator(api_key=api_key, model=llm_model)
        
        self.evaluator = Evaluator()
        
        # State tracking
        self.processed_pdfs = []
        print("GraphRAG system initialized successfully")
    
    def process_pdf(self, pdf_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a PDF document and add it to the knowledge base.
        
        Args:
            pdf_path (str): Path to the PDF file
            doc_id (Optional[str]): Optional document ID
            
        Returns:
            Dict: Information about the processed document
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Use filename as doc_id if not provided
        if doc_id is None:
            doc_id = os.path.basename(pdf_path).replace(".pdf", "")
        
        # 1. Extract text and split into chunks
        chunk_data = self.chunker.process_pdf(pdf_path)
        chunks = chunk_data['chunks']
        print(f"Extracted {len(chunks)} chunks from PDF")
        
        # 2. Generate embeddings for chunks
        chunk_data = self.embedder.embed_chunks(chunk_data)
        embeddings = chunk_data['embeddings']
        
        # 3. Store embeddings in vector database
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.vector_db.create_index(embeddings, chunks, chunk_ids)
        
        # 4. Build knowledge graph
        self.graph_builder.build_graph_from_chunks(chunks, chunk_ids)
        
        # 5. Save state
        self.processed_pdfs.append({
            'path': pdf_path,
            'doc_id': doc_id,
            'chunk_count': len(chunks)
        })
        
        # Return details about the processed document
        return {
            'document_id': doc_id,
            'path': pdf_path,
            'chunks': len(chunks),
            'embedding_dim': embeddings.shape[1],
            'status': 'processed'
        }
    
    def answer_question(self, question: str, top_k_vector: int = 3, top_k_graph: int = 2) -> Dict[str, Any]:
        """
        Answer a question using the GraphRAG pipeline.
        
        Args:
            question (str): The question to answer
            top_k_vector (int): Number of chunks to retrieve via vector search
            top_k_graph (int): Number of additional chunks to retrieve via graph
            
        Returns:
            Dict: The answer and retrieval information
        """
        print(f"Answering question: {question}")
        
        # 1. Embed the question
        query_embedding = self.embedder.embed_query(question)
        
        # 2. Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve_chunks(
            query_vector=query_embedding,
            top_k_vector=top_k_vector,
            top_k_graph=top_k_graph
        )
        
        # 3. Generate answer using retrieved chunks
        if self.generator is None:
            answer = {
                'answer': "Cannot generate answer: OpenAI API key not provided.",
                'model': "none"
            }
        else:
            answer = self.generator.generate_answer(
                query=question,
                context_chunks=retrieval_results['all_chunks']
            )
        
        # 4. Combine results
        result = {
            'question': question,
            'answer': answer['answer'],
            'model': answer.get('model', "unknown"),
            'retrieval': {
                'vector_chunks': len(retrieval_results['vector_chunks']),
                'graph_chunks': len(retrieval_results['graph_chunks']),
                'total_chunks': len(retrieval_results['all_chunks'])
            }
        }
        
        if 'usage' in answer:
            result['token_usage'] = answer['usage']
        
        return result
    
    def save_state(self, base_path: str) -> None:
        """
        Save the current state of the GraphRAG system.
        
        Args:
            base_path (str): Base path to save state files
        """
        # Ensure directories exist
        os.makedirs(base_path, exist_ok=True)
        
        # Save vector index
        self.vector_db.save_index(os.path.join(base_path, "vector_index"))
        
        # Save knowledge graph
        self.graph_builder.save_graph(os.path.join(base_path, "knowledge_graph.json"))
        
        # Save graph visualization
        self.graph_builder.visualize_graph(os.path.join(base_path, "graph_visualization.png"))
        
        # Save processed PDF info
        with open(os.path.join(base_path, "processed_pdfs.json"), 'w') as f:
            json.dump(self.processed_pdfs, f, indent=2)
        
        print(f"GraphRAG state saved to {base_path}")
    
    def load_state(self, base_path: str) -> None:
        """
        Load a previously saved state of the GraphRAG system.
        
        Args:
            base_path (str): Base path where state files are saved
        """
        # Load vector index
        self.vector_db.load_index(os.path.join(base_path, "vector_index"))
        
        # Load knowledge graph
        self.graph_builder.load_graph(os.path.join(base_path, "knowledge_graph.json"))
        
        # Load processed PDF info
        with open(os.path.join(base_path, "processed_pdfs.json"), 'r') as f:
            self.processed_pdfs = json.load(f)
        
        print(f"GraphRAG state loaded from {base_path}")


def main():
    """Main function to run the GraphRAG system from command line."""
    parser = argparse.ArgumentParser(description="GraphRAG: Knowledge Graph-enhanced RAG System")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process PDF command
    process_parser = subparsers.add_parser("process", help="Process a PDF document")
    process_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    
    # Answer question command
    answer_parser = subparsers.add_parser("answer", help="Answer a question")
    answer_parser.add_argument("question", type=str, help="The question to answer")
    answer_parser.add_argument("--vector-k", type=int, default=3, help="Number of vector chunks to retrieve")
    answer_parser.add_argument("--graph-k", type=int, default=2, help="Number of graph chunks to retrieve")
    
    # Save state command
    save_parser = subparsers.add_parser("save", help="Save system state")
    save_parser.add_argument("save_path", type=str, help="Path to save state")
    
    # Load state command
    load_parser = subparsers.add_parser("load", help="Load system state")
    load_parser.add_argument("load_path", type=str, help="Path to load state from")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate system performance")
    evaluate_parser.add_argument("eval_file", type=str, help="JSON file with evaluation data")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create directories
    ensure_directories_exist()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY") and args.command in ["answer", "evaluate"]:
        print("WARNING: OPENAI_API_KEY environment variable not set. Set it in .env file.")
        if args.command == "answer":
            print("Answer generation will not work without an API key.")
    
    # Initialize GraphRAG system
    graphrag = GraphRAG()
    
    # Execute command
    if args.command == "process":
        result = graphrag.process_pdf(args.pdf_path)
        print(json.dumps(result, indent=2))
        
        # Save state after processing
        graphrag.save_state("data/state")
        
    elif args.command == "answer":
        # Load state if exists
        if os.path.exists("data/state/vector_index.index"):
            graphrag.load_state("data/state")
        
        result = graphrag.answer_question(
            args.question,
            top_k_vector=args.vector_k,
            top_k_graph=args.graph_k
        )
        
        print("\nQuestion:")
        print(result['question'])
        print("\nAnswer:")
        print(result['answer'])
        print("\nRetrieval Stats:")
        print(f"- Vector chunks: {result['retrieval']['vector_chunks']}")
        print(f"- Graph chunks: {result['retrieval']['graph_chunks']}")
        print(f"- Total chunks: {result['retrieval']['total_chunks']}")
        
        if 'token_usage' in result:
            print("\nToken Usage:")
            print(f"- Prompt tokens: {result['token_usage']['prompt_tokens']}")
            print(f"- Completion tokens: {result['token_usage']['completion_tokens']}")
            print(f"- Total tokens: {result['token_usage']['total_tokens']}")
        
    elif args.command == "save":
        graphrag.save_state(args.save_path)
        
    elif args.command == "load":
        graphrag.load_state(args.load_path)
        print("System state loaded successfully.")
        
    elif args.command == "evaluate":
        # Load state if exists
        if os.path.exists("data/state/vector_index.index"):
            graphrag.load_state("data/state")
        
        # Load evaluation data
        with open(args.eval_file, 'r') as f:
            eval_data = json.load(f)
        
        references = [item['reference_answer'] for item in eval_data]
        questions = [item['question'] for item in eval_data]
        
        # Generate answers
        generated_answers = []
        for question in questions:
            result = graphrag.answer_question(question)
            generated_answers.append(result['answer'])
            print(f"Processed question: {question[:30]}...")
        
        # Run evaluation
        eval_results = graphrag.evaluator.evaluate(references, generated_answers)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"- BLEU-1: {eval_results['bleu1']:.4f}")
        print(f"- BLEU-4: {eval_results['bleu4']:.4f}")
        print(f"- F1 Score: {eval_results['f1']:.4f}")
        print(f"- Exact Match: {eval_results['exact_match']:.4f}")
        print(f"- Sample Count: {eval_results['sample_count']}")
        
        # Save detailed results
        with open("data/results/evaluation_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Detailed evaluation results saved to data/results/evaluation_results.json")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
