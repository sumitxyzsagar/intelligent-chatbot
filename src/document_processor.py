from typing import List, Dict, Set, Optional, Tuple
import os
import PyPDF2
import docx
import tiktoken
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from datetime import datetime
import logging
import warnings
from src.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# Suppress warnings about the huggingface_hub import
warnings.filterwarnings('ignore', category=UserWarning)

# Add compatibility layer for huggingface_hub
try:
    from huggingface_hub import hf_hub_download as cached_download
except ImportError:
    from huggingface_hub import cached_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    A comprehensive document processing system that handles multiple document formats,
    performs text extraction, chunking, and embedding for efficient semantic search.
    """
    
    # Define supported file extensions
    SUPPORTED_EXTENSIONS: Set[str] = {'.pdf', '.docx', '.txt'}
    
    def __init__(self, 
                 embedding_model: str = EMBEDDING_MODEL,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor with specified parameters.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        logger.info(f"Initializing DocumentProcessor with model: {embedding_model}")
        
        # Initialize the embedding model for semantic search
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize text splitter with specified parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize storage for processed documents
        self.index = None  # FAISS index for vector similarity search
        self.chunks = []   # Store text chunks for retrieval
        self.document_map = {}  # Map chunks to their source documents
        
        # Statistics for processed documents
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'processing_time': 0
        }

    def get_documents_from_folder(self, folder_path: str) -> List[str]:
        """
        Recursively scan a folder and return paths of all supported documents.
        
        Args:
            folder_path: Path to the folder containing documents
            
        Returns:
            List of file paths for supported documents
            
        Raises:
            FileNotFoundError: If the folder doesn't exist
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        document_paths = []
        
        # Walk through the folder and its subfolders
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Check if the file extension is supported
                if os.path.splitext(file)[1].lower() in self.SUPPORTED_EXTENSIONS:
                    document_paths.append(file_path)
                    
        return sorted(document_paths)  # Sort for consistent processing order

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try alternative encodings if UTF-8 fails
            encodings = ['latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Unable to decode file {file_path} with any supported encoding")

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from various document formats.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content as string
            
        Raises:
            ValueError: If file format is unsupported
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def process_documents(self, file_paths: List[str]) -> None:
        """
        Process multiple documents to create searchable embeddings.
        
        Args:
            file_paths: List of paths to document files
        """
        start_time = datetime.now()
        all_text = []
        processed_files = []
        
        # Extract text from all documents
        for file_path in file_paths:
            try:
                logger.info(f"Processing document: {os.path.basename(file_path)}")
                text = self.extract_text(file_path)
                all_text.append(text)
                processed_files.append(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        
        # Split text into chunks
        self.chunks = []
        for idx, text in enumerate(all_text):
            chunks = self.text_splitter.split_text(text)
            self.chunks.extend(chunks)
            
            # Map chunks to source documents
            chunk_start = len(self.chunks) - len(chunks)
            for i in range(len(chunks)):
                self.document_map[chunk_start + i] = processed_files[idx]
        
        # Create embeddings for all chunks
        if self.chunks:
            embeddings = self.embedding_model.encode(self.chunks)
            
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))
        
        # Update statistics
        self.stats['total_documents'] = len(processed_files)
        self.stats['total_chunks'] = len(self.chunks)
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Processed {len(processed_files)} documents into {len(self.chunks)} chunks")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")

    def process_folder(self, folder_path: str, recursive: bool = True) -> None:
        """
        Process all supported documents in a folder.
        
        Args:
            folder_path: Path to the folder containing documents
            recursive: Whether to process documents in subfolders
        """
        document_paths = self.get_documents_from_folder(folder_path)
        
        if not document_paths:
            logger.warning(f"No supported documents found in {folder_path}")
            return
            
        logger.info(f"Found {len(document_paths)} documents to process:")
        for path in document_paths:
            logger.info(f"- {os.path.basename(path)}")
            
        self.process_documents(document_paths)

    def find_relevant_chunks(self, 
                           query: str, 
                           k: int = 3, 
                           include_sources: bool = True) -> List[Tuple[str, str]]:
        """
        Find the most relevant chunks for a given query.
        
        Args:
            query: Search query string
            k: Number of chunks to retrieve
            include_sources: Whether to include source document paths
            
        Returns:
            List of tuples containing (chunk_text, source_document)
        """
        if not self.index or not self.chunks:
            logger.warning("No documents have been processed yet")
            return []
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        results = []
        for idx in indices[0]:
            chunk = self.chunks[idx]
            source = self.document_map.get(idx, "Unknown source") if include_sources else None
            results.append((chunk, source))
            
        return results

    def get_context(self, query: str, max_chunks: int = 3) -> str:
        """
        Get relevant context for a query by combining relevant chunks.
        
        Args:
            query: Search query string
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Combined context string from relevant chunks
        """
        relevant_chunks = self.find_relevant_chunks(query, k=max_chunks)
        return "\n\n".join(chunk for chunk, _ in relevant_chunks)

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
    
__all__ = ['DocumentProcessor']