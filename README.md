# Document Q&A Assistant

A Streamlit-based application that uses RAG (Retrieval-Augmented Generation) to answer questions about your PDF documents. The application processes your documents, creates embeddings, and uses them to provide accurate answers with source citations.

## Features

- Upload and process PDF documents
- Interactive chat interface
- Source citations with viewable context
- Hybrid search for better document retrieval
- Error handling and progress tracking

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simple_doc_chat.git
   cd simple_doc_chat
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your PDF documents in the `data/` directory

2. Build the document index:
   ```bash
   python src/build_index.py
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

4. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Project Structure

```
simple_doc_chat/
├── data/               # Directory for PDF documents
├── src/               # Source code
│   ├── app.py         # Main Streamlit application
│   └── build_index.py # Document indexing script
├── tests/             # Unit tests
├── .env              # Environment variables (create this file)
├── .gitignore        # Git ignore rules
└── requirements.txt   # Python dependencies
```

## Configuration

Create a `.env` file in the root directory with your configuration:
```env
# Example configuration
OPENAI_API_KEY=your_api_key_here
```

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Code Style

The code follows PEP 8 guidelines and includes comprehensive docstrings.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 