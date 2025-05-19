
"""
Streamlit web application for GraphRAG.
This provides a user interface for uploading PDFs, asking questions, and visualizing results.
"""

import os
import json
import tempfile
import numpy as np
import streamlit as st
from PIL import Image

# Import the GraphRAG system
from main import GraphRAG, ensure_directories_exist

# Ensure necessary directories exist
ensure_directories_exist()

# Set page title and favicon
st.set_page_config(
    page_title="GraphRAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'graphrag' not in st.session_state:
    st.session_state.graphrag = GraphRAG()

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None

if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False

# Helper functions
def process_pdf(uploaded_file):
    """Process an uploaded PDF file."""
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        # Process the PDF with GraphRAG
        result = st.session_state.graphrag.process_pdf(tmp_path, doc_id=uploaded_file.name)
        
        # Add to processed files
        st.session_state.processed_files.append(uploaded_file.name)
        
        # Save state
        st.session_state.graphrag.save_state("data/state")
        
        return result
    finally:
        # Remove the temporary file
        os.unlink(tmp_path)

def answer_question(question):
    """Answer a question with GraphRAG."""
    result = st.session_state.graphrag.answer_question(
        question,
        top_k_vector=st.session_state.get('vector_k', 3),
        top_k_graph=st.session_state.get('graph_k', 2)
    )
    
    # Visualize graph if needed
    if st.session_state.show_graph:
        st.session_state.graphrag.graph_builder.visualize_graph("data/temp_graph.png")
    
    return result

# Main app header
st.title("📚 GraphRAG Q&A System")
st.markdown("Knowledge graph-enhanced retrieval for research papers")

# Sidebar
with st.sidebar:
    st.header("Settings & Options")
    
    # OpenAI API Key
    api_key = st.text_input("OpenAI API Key", type="password", 
                          help="Enter your OpenAI API key to enable answer generation",
                          value=os.getenv("OPENAI_API_KEY", ""))
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        # Reinitialize generator if needed
        if st.session_state.graphrag.generator is None:
            st.session_state.graphrag.generator = st.session_state.graphrag.Generator(api_key=api_key)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        st.slider("Vector chunks (k)", min_value=1, max_value=10, value=3, key="vector_k")
        st.slider("Graph chunks (k)", min_value=0, max_value=10, value=2, key="graph_k")
        st.checkbox("Show knowledge graph", key="show_graph")
    
    # File management
    with st.expander("Processed Files", expanded=True):
        if not st.session_state.processed_files:
            st.info("No files processed yet.")
        else:
            for file in st.session_state.processed_files:
                st.text(f"• {file}")
    
    # State management
    with st.expander("System State"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save State", use_container_width=True):
                st.session_state.graphrag.save_state("data/state")
                st.success("State saved!")
        
        with col2:
            if st.button("Load State", use_container_width=True):
                try:
                    st.session_state.graphrag.load_state("data/state")
                    st.success("State loaded!")
                except:
                    st.error("No saved state found.")

# Main content
tabs = st.tabs(["📄 Upload Papers", "❓ Ask Questions", "📊 Results"])

# Upload Papers tab
with tabs[0]:
    st.header("Upload Research Papers")
    
    uploaded_file = st.file_uploader("Upload a PDF research paper", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                result = process_pdf(uploaded_file)
            
            st.success(f"Successfully processed {uploaded_file.name}")
            st.json(result)

# Ask Questions tab
with tabs[1]:
    st.header("Ask Questions")
    
    if not st.session_state.processed_files:
        st.warning("Please upload and process at least one PDF file first.")
    else:
        question = st.text_input("Enter your research question", key="question_input")
        
        if st.button("Get Answer", use_container_width=True):
            st.session_state.current_question = question
            
            with st.spinner("Generating answer..."):
                result = answer_question(question)
                st.session_state.current_answer = result
            
            st.success("Answer generated!")

# Results tab
with tabs[2]:
    if st.session_state.current_answer:
        st.header("Answer")
        
        # Show the question and answer
        st.subheader("Question")
        st.write(st.session_state.current_question)
        
        st.subheader("Answer")
        st.write(st.session_state.current_answer['answer'])
        
        # Show retrieval statistics
        st.subheader("Retrieval Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Vector Chunks", st.session_state.current_answer['retrieval']['vector_chunks'])
        col2.metric("Graph Chunks", st.session_state.current_answer['retrieval']['graph_chunks'])
        col3.metric("Total Chunks", st.session_state.current_answer['retrieval']['total_chunks'])
        
        # Show token usage if available
        if 'token_usage' in st.session_state.current_answer:
            st.subheader("Token Usage")
            usage = st.session_state.current_answer['token_usage']
            col1, col2, col3 = st.columns(3)
            col1.metric("Prompt Tokens", usage['prompt_tokens'])
            col2.metric("Completion Tokens", usage['completion_tokens'])
            col3.metric("Total Tokens", usage['total_tokens'])
        
        # Show graph visualization if enabled
        if st.session_state.show_graph and os.path.exists("data/temp_graph.png"):
            st.subheader("Knowledge Graph Visualization")
            st.image("data/temp_graph.png")
    else:
        st.info("Ask a question to see results here.")

# Footer
st.markdown("---")
st.markdown(
    "GraphRAG Q&A System - A knowledge graph-enhanced retrieval-augmented generation system"
)
