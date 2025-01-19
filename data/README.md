# Data Directory

This directory is where you should place your PDF documents for the Document Q&A Assistant to process.

## Usage

1. Place any PDF documents you want to query in this directory
2. Run the indexing script:
   ```bash
   python src/build_index.py
   ```
3. The documents will be processed and indexed in the database

## Supported Formats

Currently, the application supports:
- PDF documents (*.pdf)

## Notes

- Make sure your PDF documents are text-based (not scanned images)
- The indexing process may take some time depending on the size and number of documents
- The indexed data will be stored in `raglite.db` in the root directory 