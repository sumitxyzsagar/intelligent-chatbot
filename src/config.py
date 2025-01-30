import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model and processing configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Lightweight but effective embedding model
CHUNK_SIZE = 500  # Size of text chunks for processing
CHUNK_OVERLAP = 50  # Overlap between chunks to maintain context

# Conversation settings
MAX_HISTORY = 3  # Number of conversation exchanges to remember
TEMPERATURE = 0.7  # Controls response randomness (0.0-1.0)
MAX_TOKENS = 500  # Maximum length of generated responses

# API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Get API key from environment variables

# Error messages
ERROR_MESSAGES = {
    'api_error': 'Error connecting to OpenAI API: {}',
    'file_error': 'Error processing file: {}',
    'invalid_format': 'Unsupported file format: {}'
}