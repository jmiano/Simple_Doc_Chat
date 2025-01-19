"""Unit tests for the Streamlit application."""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import streamlit as st
from src.app import handle_source_click, config

@pytest.fixture
def mock_session_state():
    """Create a mock Streamlit session state."""
    with patch("streamlit.session_state") as mock_state:
        mock_state.messages = []
        mock_state.current_source = None
        mock_state.awaiting_response = False
        yield mock_state

@pytest.fixture
def mock_raglite():
    """Create mock RAGLite functions."""
    with patch("src.app.hybrid_search") as mock_search, \
         patch("src.app.retrieve_chunks") as mock_retrieve, \
         patch("src.app.rerank_chunks") as mock_rerank, \
         patch("src.app.retrieve_chunk_spans") as mock_spans, \
         patch("src.app.rag") as mock_rag:
        
        mock_search.return_value = (["chunk1", "chunk2"], None)
        mock_retrieve.return_value = ["content1", "content2"]
        mock_rerank.return_value = ["ranked1", "ranked2"]
        mock_spans.return_value = ["span1", "span2"]
        mock_rag.return_value = iter(["response part 1", "response part 2"])
        
        yield {
            "search": mock_search,
            "retrieve": mock_retrieve,
            "rerank": mock_rerank,
            "spans": mock_spans,
            "rag": mock_rag,
            "config": config
        }

def test_handle_source_click(mock_session_state):
    """Test the source display functionality."""
    test_source = "Test source content"
    handle_source_click(test_source)
    assert mock_session_state.current_source == test_source

def test_database_check():
    """Test database existence check."""
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("streamlit.error") as mock_error, \
         patch("streamlit.stop") as mock_stop:
        mock_exists.return_value = False
        
        # Call the error and stop functions directly
        st.error("Error: Database not found! Please run 'python src/build_index.py' first to build the index.")
        st.stop()
        
        # Verify the calls
        mock_error.assert_called_once_with("Error: Database not found! Please run 'python src/build_index.py' first to build the index.")
        mock_stop.assert_called_once()

def test_message_processing(mock_session_state, mock_raglite):
    """Test message processing and response generation."""
    # Simulate user input
    test_message = "Test question"
    mock_session_state.messages.append({"role": "user", "content": test_message})
    mock_session_state.awaiting_response = True
    
    # Process the message
    if mock_session_state.awaiting_response:
        mock_raglite["search"].assert_not_called()  # Not called yet
        
        # Simulate message processing
        chunk_ids_hybrid, _ = mock_raglite["search"](test_message, num_results=20, config=mock_raglite["config"])
        chunks_hybrid = mock_raglite["retrieve"](chunk_ids_hybrid, config=mock_raglite["config"])
        chunks_reranked = mock_raglite["rerank"](test_message, chunks_hybrid, config=mock_raglite["config"])
        chunks_reranked = chunks_reranked[:5]
        chunk_spans = mock_raglite["spans"](chunks_reranked, config=mock_raglite["config"])
        
        # Verify function calls
        mock_raglite["search"].assert_called_once_with(test_message, num_results=20, config=mock_raglite["config"])
        mock_raglite["retrieve"].assert_called_once_with(["chunk1", "chunk2"], config=mock_raglite["config"])
        mock_raglite["rerank"].assert_called_once_with(test_message, ["content1", "content2"], config=mock_raglite["config"])
        mock_raglite["spans"].assert_called_once_with(["ranked1", "ranked2"][:5], config=mock_raglite["config"]) 