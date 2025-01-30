# Intelligent Document-Aware Chatbot

A lightweight, intelligent chatbot system that can be trained on custom documents and generate contextual responses. This chatbot uses modern NLP techniques including document embedding and semantic search to provide relevant, context-aware responses.

## Features

- Document Processing:
  - Support for multiple document formats (PDF, DOCX, TXT)
  - Intelligent text chunking with context preservation
  - Efficient document embedding using sentence transformers
  - Fast similarity search using FAISS

- Conversation Management:
  - Context-aware responses using OpenAI's GPT-3.5
  - Conversation history management
  - Configurable response parameters
  - Error handling and recovery

## Project Structure

```
intelligent_chatbot/
│
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
├── README.md          # Project documentation
├── data/             # Document storage
│   └── sample.txt    # Sample training document
│
├── src/              # Source code
│   ├── __init__.py
│   ├── document_processor.py  # Document processing logic
│   ├── chatbot.py            # Chatbot implementation
│   └── config.py             # Configuration settings
│
└── main.py           # Application entry point
```

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your OpenAI API key:
   - Copy the `.env` file
   - Add your OpenAI API key to the `.env` file
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Add your training documents:
   - Place your documents in the `data/` directory
   - Supported formats: PDF, DOCX, TXT
   - Update document paths in `main.py` if needed