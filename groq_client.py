import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GroqClient:
    def __init__(self, api_key=None, model=None):
        """
        Initialize Groq client
        
        Args:
            api_key: Optional API key (defaults to GROQ_API_KEY from .env)
            model: Optional model name (defaults to GROQ_MODEL from .env or llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
        
        self.client = Groq(api_key=self.api_key)
    
    def chat(self, messages, temperature=1, max_tokens=1024, stream=False):
        """
        Send a chat completion request to Groq
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Controls randomness (0-2)
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            
        Returns:
            Response from Groq API
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            return response
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            raise
    
    def simple_chat(self, user_message, system_message=None, temperature=1, max_tokens=1024):
        """
        Simplified chat method for single message exchanges
        
        Args:
            user_message: The user's message
            system_message: Optional system message to set context
            temperature: Controls randomness (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            String response from the model
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return response.choices[0].message.content
    
    def stream_chat(self, messages, temperature=1, max_tokens=1024):
        """
        Stream a chat completion response
        
        Args:
            messages: List of message dictionaries
            temperature: Controls randomness (0-2)
            max_tokens: Maximum tokens in response
            
        Yields:
            Response chunks from Groq API
        """
        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content