"""
Bank Statement Metadata Extractor
"""
import json
import pandas as pd
from typing import Dict, Optional
# Change these imports:
from llm_providers.llm_provider_interface import LLMProvider
from prompts.prompt_templates import METADATA_EXTRACTION_PROMPT

class MetadataExtractor:
    """Extracts and saves bank account metadata"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
    
    def extract_metadata(self, pdf_data: bytes) -> Optional[Dict]:
        """Extract account metadata from PDF"""
        print("\n📋 Extracting account metadata...")
        
        try:
            response_text = self.llm_provider.extract_metadata(
                pdf_data=pdf_data,
                prompt=METADATA_EXTRACTION_PROMPT,
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse JSON response
            metadata = self._parse_json(response_text)
            
            if metadata:
                print(f" ✅ Extracted metadata for account: {metadata.get('account_number', 'Unknown')}")
                return metadata
            else:
                print(" ⚠️ Failed to parse metadata")
                return None
                
        except Exception as e:
            print(f" ❌ Metadata extraction error: {e}")
            return None
    
    def save_metadata(self, metadata: Dict, output_path: str):
        """Save metadata to JSON file"""
        if not metadata:
            print(" ⚠️ No metadata to save")
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f" 💾 Metadata saved to: {output_path}")
        except Exception as e:
            print(f" ❌ Error saving metadata: {e}")
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        # Remove markdown code blocks
        text = text.replace('``````', '').strip()
        
        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Try extracting object by braces
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except:
            pass
        
        return None
