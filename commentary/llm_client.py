"""
LLM client for commentary service.
Supports Ollama and LM Studio HTTP APIs.
"""
import requests
import json
import time
from typing import Dict, List, Optional
import os

class LLMClient:
    """
    Client for interacting with local LLM services like Ollama or LM Studio.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b-instruct-fp16"):
        self.base_url = base_url
        self.model_name = model_name
        self.session = requests.Session()
        
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            bool: True if service is available
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
            
    def generate_commentary(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate commentary text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated commentary text
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            # Send request to LLM
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"LLM request failed with status {response.status_code}")
                return "Sorry, I couldn't generate commentary at this moment."
                
        except Exception as e:
            print(f"Error generating commentary: {e}")
            return "Sorry, I couldn't generate commentary at this moment."

class OllamaClient(LLMClient):
    """
    Specialized client for Ollama service.
    """
    
    def __init__(self, model_name: str = "llama3"):
        super().__init__(base_url="http://localhost:11434", model_name=model_name)

class LMStudioClient(LLMClient):
    """
    Specialized client for LM Studio service.
    """
    
    def __init__(self, model_name: str = "llama3"):
        super().__init__(base_url="http://localhost:1234", model_name=model_name)

# Factory function to create LLM client
def create_llm_client(client_type: str = "ollama", model_name: str = "llama3") -> LLMClient:
    """
    Create an LLM client based on the specified type.
    
    Args:
        client_type: Type of client ("ollama" or "lmstudio")
        model_name: Name of the model to use
        
    Returns:
        LLMClient: Initialized client
    """
    if client_type.lower() == "ollama":
        return OllamaClient(model_name)
    elif client_type.lower() == "lmstudio":
        return LMStudioClient(model_name)
    else:
        # Default to Ollama
        return OllamaClient(model_name)

# Test function to verify LLM connectivity
def test_llm_connection():
    """Test if LLM service is available."""
    client = create_llm_client()
    
    if client.is_available():
        print("LLM service is available")
        return True
    else:
        print("LLM service is not available")
        return False

if __name__ == "__main__":
    # Test LLM connection
    test_llm_connection()
