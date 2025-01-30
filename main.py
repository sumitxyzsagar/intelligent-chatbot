import os
import sys
import logging
import argparse
from typing import List, Optional
from datetime import datetime
from src.document_processor import DocumentProcessor
from src.chatbot import IntelligentChatbot
from dotenv import load_dotenv

# Add the project root to Python path for proper imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()

# Configure logging with both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

class ChatbotApplication:
    """
    Main application class that coordinates the document processor and chatbot.
    Handles user interaction, command processing, and system initialization.
    """
    
    def __init__(self, docs_folder: str = 'data', verbose: bool = False):
        """
        Initialize the chatbot application with specified parameters.
        
        Args:
            docs_folder: Path to the folder containing training documents
            verbose: Whether to print detailed processing information
        """
        self.docs_folder = docs_folder
        self.verbose = verbose
        self.chatbot = None
        self.doc_processor = None
        self.start_time = datetime.now()

    def validate_environment(self) -> bool:
        """
        Validate the environment configuration and requirements.
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        try:
            # Check if documents folder exists
            if not os.path.exists(self.docs_folder):
                logger.error(f"Documents folder not found: {self.docs_folder}")
                print(f"\nError: Documents folder '{self.docs_folder}' does not exist.")
                print("Please create the folder and add your documents before running the chatbot.")
                return False
            
            # Check if OpenAI API key is set
            if not os.getenv('OPENAI_API_KEY'):
                logger.error("OPENAI_API_KEY environment variable is not set")
                print("\nError: OpenAI API key not found!")
                print("Please set your OpenAI API key in the .env file or environment variables.")
                return False
            
            # Check if documents folder contains supported files
            has_documents = False
            supported_extensions = {'.pdf', '.docx', '.txt'}
            for root, _, files in os.walk(self.docs_folder):
                if any(f.lower().endswith(tuple(supported_extensions)) for f in files):
                    has_documents = True
                    break
            
            if not has_documents:
                logger.warning("No supported documents found in documents folder")
                print(f"\nWarning: No supported documents found in '{self.docs_folder}'")
                print("Please add PDF, DOCX, or TXT files to process.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Environment validation error: {str(e)}")
            return False

    def initialize_system(self) -> bool:
        """
        Initialize the document processor and chatbot components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize document processor
            print("\nInitializing document processor...")
            self.doc_processor = DocumentProcessor()
            
            # Process all documents in the folder
            start_time = datetime.now()
            self.doc_processor.process_folder(self.docs_folder)
            
            # Calculate and display processing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            stats = self.doc_processor.get_statistics()
            
            if self.verbose:
                print("\nDocument processing statistics:")
                print(f"- Total documents processed: {stats['total_documents']}")
                print(f"- Total text chunks created: {stats['total_chunks']}")
                print(f"- Processing time: {processing_time:.2f} seconds")
            
            # Initialize chatbot with processed documents
            print("\nInitializing chatbot...")
            self.chatbot = IntelligentChatbot(self.doc_processor)
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            print(f"\nError during initialization: {str(e)}")
            return False

    def display_welcome_message(self) -> None:
        """Display welcome message and usage instructions."""
        print("\n=" * 50)
        print("Welcome to the Intelligent Document Chatbot!")
        print("=" * 50)
        print("\nThis chatbot can answer questions about your documents.")
        print("\nAvailable commands:")
        print("- 'quit': Exit the application")
        print("- 'clear': Clear conversation history")
        print("- 'stats': Show document processing statistics")
        print("- 'help': Show this help message")
        print("\nStart asking questions about your documents!")
        print("-" * 50)

    def handle_special_commands(self, command: str) -> bool:
        """
        Handle special system commands like quit, clear, stats, etc.
        
        Args:
            command: User input command
            
        Returns:
            bool: True if command was handled, False otherwise
        """
        command = command.lower().strip()
        
        if command == 'quit':
            print("\nGoodbye! Thank you for using the chatbot.")
            return True
            
        elif command == 'clear':
            if self.chatbot:
                self.chatbot.clear_history()
                print("\nConversation history cleared.")
            return True
            
        elif command == 'stats':
            if self.doc_processor:
                stats = self.doc_processor.get_statistics()
                print("\nDocument Processing Statistics:")
                print(f"Total documents processed: {stats['total_documents']}")
                print(f"Total text chunks: {stats['total_chunks']}")
                print(f"Processing time: {stats['processing_time']:.2f} seconds")
                print(f"System uptime: {(datetime.now() - self.start_time).total_seconds():.0f} seconds")
            return True
            
        elif command == 'help':
            self.display_welcome_message()
            return True
            
        return False

    def run(self) -> None:
        """Run the main chatbot application loop."""
        try:
            # Validate environment
            if not self.validate_environment():
                return
            
            # Initialize system components
            if not self.initialize_system():
                return
            
            # Display welcome message
            self.display_welcome_message()
            
            # Main interaction loop
            while True:
                try:
                    # Get user input
                    user_input = input("\nYou: ").strip()
                    
                    # Skip empty input
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if self.handle_special_commands(user_input):
                        if user_input.lower() == 'quit':
                            break
                        continue
                    
                    # Generate and display response
                    response = self.chatbot.generate_response(user_input)
                    print(f"\nBot: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Exiting...")
                    break
                    
                except Exception as e:
                    logger.error(f"Error processing input: {str(e)}")
                    print(f"\nError processing your input: {str(e)}")
                    print("Please try again or type 'quit' to exit.")
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")
            print("Please check the logs for more information.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Intelligent Document Chatbot')
    parser.add_argument('--docs-folder', type=str, default='data',
                      help='Path to the folder containing documents (default: data)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Create and run the chatbot application
        app = ChatbotApplication(docs_folder=args.docs_folder, verbose=args.verbose)
        app.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()