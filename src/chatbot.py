from typing import List, Dict, Optional
from openai import OpenAI
import os
from src.config import OPENAI_API_KEY, MAX_HISTORY, TEMPERATURE, MAX_TOKENS
from src.document_processor import DocumentProcessor

class IntelligentChatbot:
    def __init__(self, document_processor: DocumentProcessor):
        """
        Initialize the chatbot with document processor and OpenAI client.
        
        Args:
            document_processor: Initialized DocumentProcessor instance
        """
        self.document_processor = document_processor
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.conversation_history = []
        
        # Define system prompt for consistent behavior
        self.system_prompt = """You are a helpful assistant with access to specific documentation. 
        Always base your responses on the provided context. If you're unsure or the information 
        isn't in the context, say so clearly. Use a friendly, professional tone."""
    
    def generate_response(self, query: str) -> str:
        """
        Generate a response based on the query and document context.
        
        Args:
            query: User's question or input
            
        Returns:
            Generated response string
        """
        # Get relevant context from documents
        context = self.document_processor.get_context(query)
        
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Context: {context}"}
        ]
        
        # Add relevant conversation history for context
        for msg in self.conversation_history[-MAX_HISTORY:]:
            messages.append(msg)
            
        # Add current query
        messages.append({"role": "user", "content": query})
        
        try:
            # Generate response using OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            # Extract and store response
            bot_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": bot_response}
            ])
            
            return bot_response
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []