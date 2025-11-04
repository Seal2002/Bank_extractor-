"""
LLM Provider Interface - Abstract base class for all LLM providers
This allows easy switching between different LLM APIs (Gemini, OpenAI, Claude, etc.)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
    
    @abstractmethod
    def extract_transactions(
        self, 
        pdf_data: bytes, 
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 30000
    ) -> str:
        """Extract transactions from PDF using the LLM"""
        pass
    
    @abstractmethod
    def extract_metadata(
        self,
        pdf_data: bytes,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """Extract account metadata from PDF"""
        pass
