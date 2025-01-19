"""
A Streamlit-based Document Q&A Assistant that uses RAG (Retrieval-Augmented Generation)
to answer questions about PDF documents. The app uses raglite for document processing
and retrieval, and provides an interactive chat interface with source citations.
"""

import os
import streamlit as st
from pathlib import Path
from raglite import (
    RAGLiteConfig, rag, hybrid_search, retrieve_chunks,
    rerank_chunks, retrieve_chunk_spans, create_rag_instruction
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure RAGLite with database and model settings
config = RAGLiteConfig(
    db_url="sqlite:///raglite.db",
    llm="gpt-4o-mini",  # Large language model for generating responses
    embedder="text-embedding-3-large"  # Model for creating document embeddings
)

# Initialize Streamlit session state variables for maintaining chat state
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Store chat history
if 'current_source' not in st.session_state:
    st.session_state.current_source = None  # Currently displayed source document
if 'awaiting_response' not in st.session_state:
    st.session_state.awaiting_response = False  # Flag to prevent multiple simultaneous requests

# Set up the main layout
st.title("Document Q&A Assistant")

# Verify database existence before proceeding
if not Path("raglite.db").exists():
    st.error("Error: Database not found! Please run 'python src/build_index.py' first to build the index.")
    st.stop()

# Create a sidebar for displaying source documents
with st.sidebar:
    st.header("Sources")
    if st.session_state.current_source:
        st.text_area("Selected Source", st.session_state.current_source, height=400)

def handle_source_click(source_text: str) -> None:
    """
    Update the sidebar to display the selected source document.
    
    Args:
        source_text (str): The text content of the selected source document
    """
    st.session_state.current_source = source_text

# Display the chat history with source citations
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            st.write("---\n**Sources:**")
            cols = st.columns(3)
            for i, source in enumerate(message["sources"]):
                if cols[i].button(f"📄 Source {i+1}", key=f"source_{message['id']}_{i}", on_click=handle_source_click, args=(source,)):
                    pass

# Handle user input
if prompt := st.chat_input("Ask a question about the documents", disabled=st.session_state.awaiting_response):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.awaiting_response = True
    st.rerun()

# Process the response when awaiting one
if st.session_state.awaiting_response:
    # Retrieve the last user message
    last_user_message = next(msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user")
    
    # Perform hybrid search to find relevant document chunks
    chunk_ids_hybrid, _ = hybrid_search(last_user_message, num_results=20, config=config)
    chunks_hybrid = retrieve_chunks(chunk_ids_hybrid, config=config)
    
    # Rerank chunks by relevance and select top results
    chunks_reranked = rerank_chunks(last_user_message, chunks_hybrid, config=config)
    chunks_reranked = chunks_reranked[:5]  # Keep top 5 most relevant chunks
    
    # Get surrounding context for the selected chunks
    chunk_spans = retrieve_chunk_spans(chunks_reranked, config=config)
    
    # Prepare the RAG instruction with context
    messages = []
    messages.append(create_rag_instruction(user_prompt=last_user_message, context=chunk_spans))
    
    # Generate the response using RAG
    response = ""
    for update in rag(messages, config=config):
        response += update
    
    # Prepare the top 3 sources for citation
    sources = []
    for i, chunk_span in enumerate(chunk_spans[:3]):
        source_text = f"Source {i+1}:\n\n{chunk_span}"
        sources.append(source_text)
    
    # Add the assistant's response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
        "id": len(st.session_state.messages)
    })
    
    # Reset the response flag and refresh the UI
    st.session_state.awaiting_response = False
    st.rerun() 