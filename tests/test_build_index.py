"""Unit tests for the document indexing functionality."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from src.build_index import build_index, config

@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a temporary data directory with test PDF files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create some dummy PDF files
    (data_dir / "test1.pdf").touch()
    (data_dir / "test2.pdf").touch()
    (data_dir / "not_a_pdf.txt").touch()
    
    return data_dir

@pytest.fixture
def mock_config():
    """Create a mock RAGLite configuration."""
    return Mock(
        db_url="sqlite:///test.db",
        llm="test-model",
        embedder="test-embedder"
    )

def test_build_index_empty_directory(tmp_path):
    """Test build_index with an empty data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    with patch("src.build_index.Path") as mock_path:
        mock_path.return_value.glob.return_value = []
        build_index()
        mock_path.return_value.glob.assert_called_once_with("*.pdf")

def test_build_index_with_files(mock_data_dir):
    """Test build_index with PDF files present."""
    with patch("src.build_index.insert_document") as mock_insert:
        with patch("src.build_index.Path") as mock_path:
            mock_path.return_value.glob.return_value = [
                mock_data_dir / "test1.pdf",
                mock_data_dir / "test2.pdf"
            ]
            build_index()
            
            # Verify insert_document was called for each PDF
            assert mock_insert.call_count == 2
            mock_insert.assert_any_call(mock_data_dir / "test1.pdf", config=config)
            mock_insert.assert_any_call(mock_data_dir / "test2.pdf", config=config)

def test_build_index_handles_errors():
    """Test build_index handles errors during document insertion."""
    with patch("src.build_index.insert_document") as mock_insert:
        with patch("src.build_index.Path") as mock_path:
            mock_path.return_value.glob.return_value = [Path("test.pdf")]
            mock_insert.side_effect = Exception("Test error")
            
            # The function should not raise an exception and should print an error
            with patch("builtins.print") as mock_print:
                try:
                    build_index()
                except Exception:
                    pytest.fail("build_index() raised an exception")
                
                mock_insert.assert_called_once_with(Path("test.pdf"), config=config)
                mock_print.assert_any_call("Processing test.pdf...") 