
# GraphRAG: Knowledge Graph-Enhanced Retrieval-Augmented Generation

This project implements a GraphRAG system - a Retrieval-Augmented Generation (RAG) pipeline enhanced with knowledge graph capabilities for improved context retrieval.

## Overview

GraphRAG combines vector-based similarity search with graph traversal to retrieve more relevant context for answering questions about research papers or other documents. The system:

1. Processes PDF documents, extracting and chunking text
2. Generates vector embeddings using sentence-transformers
3. Builds a knowledge graph by extracting entities and their relationships
4. Uses both vector similarity and graph connections for retrieval
5. Generates answers using retrieved chunks and an LLM (OpenAI)

## Project Structure

```
graphrag_project/
├── data/              # Store PDFs and intermediate data
├── src/
│   ├── chunker.py     # Extract and chunk text from PDFs
│   ├── embedder.py    # Convert text to vector embeddings
│   ├── vector_db.py   # Store and search vector embeddings with FAISS
│   ├── graph_builder.py # Build and query knowledge graph
│   ├── retriever.py   # Retrieve relevant chunks
│   ├── generator.py   # Generate answers using OpenAI
│   └── evaluator.py   # Evaluate system performance
├── main.py            # Main GraphRAG system integration
├── app.py             # Streamlit web interface
├── requirements.txt   # Dependencies
└── .env.example       # Template for environment variables
```

## Installation

1. Clone this repository
2. Install the dependencies:
```
pip install -r requirements.txt
```
3. Create a `.env` file from `.env.example` and add your OpenAI API key

## Usage

### Command Line Interface

Process a PDF document:
```
python main.py process path/to/document.pdf
```

Answer a question:
```
python main.py answer "What are the key benefits of knowledge graphs in RAG systems?"
```

Save system state:
```
python main.py save data/my_system_state
```

Load system state:
```
python main.py load data/my_system_state
```

Evaluate system performance:
```
python main.py evaluate path/to/evaluation_data.json
```

### Web Interface

Run the Streamlit app:
```
streamlit run app.py
```

The web interface allows you to:
- Upload PDF documents
- Ask questions
- View generated answers
- Visualize the knowledge graph

## Evaluation

The system includes evaluation capabilities using:
- BLEU score (for n-gram overlap)
- F1 score (for token-level precision/recall)
- Exact Match (for perfect matches)

## Technologies Used

- **Text Processing**: PyPDF2, spaCy
- **Embeddings**: sentence-transformers
- **Vector Storage**: FAISS
- **Knowledge Graph**: NetworkX
- **Answer Generation**: OpenAI API
- **Web Interface**: Streamlit

## Future Improvements

- Support for more document formats beyond PDFs
- Integration with more LLMs (local models, other APIs)
- Improved entity resolution and relationship extraction
- Advanced graph algorithms for retrieval enhancement
- Distributed processing for large document collections

## License

MIT
