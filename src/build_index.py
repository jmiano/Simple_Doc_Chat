"""
Document indexing script for the Document Q&A Assistant.
This script processes PDF documents from the data directory and builds a searchable
index using raglite's document processing capabilities.
"""

from pathlib import Path
from raglite import RAGLiteConfig, insert_document

# Configure RAGLite with database and model settings
config = RAGLiteConfig(
    db_url="sqlite:///raglite.db",
    llm="gpt-4o-mini",  # Large language model for generating responses
    embedder="text-embedding-3-large"  # Model for creating document embeddings
)

def build_index() -> None:
    """
    Build a searchable index from PDF documents in the data directory.
    
    This function:
    1. Scans the data directory for PDF files
    2. Processes each PDF file using raglite
    3. Creates embeddings and stores them in the SQLite database
    
    The resulting index is used by the main application for document retrieval
    and question answering.
    """
    # Get all PDF files from the data directory
    pdf_files = list(Path("data").glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the data directory!")
        return
    
    processed_count = 0
    errors_count = 0
    
    # Process and index each document
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        try:
            insert_document(pdf_file, config=config)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            errors_count += 1
            continue
        
    print(f"Index building complete! Processed {processed_count} documents successfully.")
    if errors_count > 0:
        print(f"Failed to process {errors_count} documents due to errors.")

if __name__ == "__main__":
    build_index() 