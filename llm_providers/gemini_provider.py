"""
Gemini LLM Provider Implementation
"""
import google.generativeai as genai
# Change this import:
from llm_providers.llm_provider_interface import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini API implementation of LLM provider"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key, model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini provider initialized with model: {model_name}")
    
    def extract_transactions(
        self,
        pdf_data: bytes,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 30000
    ) -> str:
        """Extract transactions using Gemini API"""
        pdf_part = {'mime_type': 'application/pdf', 'data': pdf_data}
        
        response = self.model.generate_content(
            [prompt, pdf_part],
            generation_config = {
                "temperature": 0,  # Deterministic extraction
                "top_p": 1.0,
                "max_output_tokens": 8192,
            }

        )
        
        return response.text.strip()
    
    def extract_metadata(
        self,
        pdf_data: bytes,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """Extract metadata using Gemini API"""
        pdf_part = {'mime_type': 'application/pdf', 'data': pdf_data}
        
        response = self.model.generate_content(
            [prompt, pdf_part],
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        
        return response.text.strip()
