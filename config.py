"""
Configuration File - Easy provider switching
"""
import os

# LLM Provider Configuration
PROVIDER_TYPE = "gemini"  # Options: "gemini", "openai", "claude", "anthropic"
API_KEY = os.getenv("GOOGLE_API_KEY")  # Change based on provider
MODEL_NAME = "gemini-2.0-flash-exp"  # Change based on provider

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "gemini": {
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash-exp",
        "class": "GeminiProvider",
        "module": "llm_providers.gemini_provider"  # FIXED: Added package name
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4-vision-preview",
        "class": "OpenAIProvider",
        "module": "llm_providers.openai_provider"  # FIXED: Added package name
    },
    # Add more providers as needed
}

# Extraction Settings
TEMPERATURE = 0.1
MAX_TOKENS_TRANSACTIONS = 30000
MAX_TOKENS_METADATA = 2000

# Processing Settings
MULTI_PAGE_THRESHOLD = 4  # Use page-by-page processing for PDFs >= 4 pages
