"""
Production Bank Statement Extractor v6.1
- Always uses page-by-page extraction (more reliable, avoids token limits)
- Modular LLM provider architecture
- Transaction ID-based deduplication
- Separate metadata extraction
"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
import PyPDF2
import importlib

from llm_providers.llm_provider_interface import LLMProvider
from core.deduplication_engine import DeduplicationEngine
from core.transaction_processor import TransactionProcessor
from core.metadata_extractor import MetadataExtractor
from prompts.prompt_templates import SINGLE_PAGE_TRANSACTION_PROMPT
import config


class BankStatementExtractor:
    """Universal bank statement extractor with modular architecture"""
    
    def __init__(self, provider_type: str = None, api_key: str = None, model_name: str = None):
        """
        Initialize extractor with specified LLM provider
        
        Args:
            provider_type: "gemini", "openai", etc. (defaults to config)
            api_key: API key for the provider (defaults to config)
            model_name: Model name (defaults to config)
        """
        # Use config defaults if not provided
        provider_type = provider_type or config.PROVIDER_TYPE
        api_key = api_key or config.API_KEY
        model_name = model_name or config.MODEL_NAME
        
        if not api_key:
            raise ValueError(f"API key not found. Set {config.PROVIDER_CONFIGS[provider_type]['api_key_env']} environment variable")
        
        # Dynamically load the provider
        self.llm_provider = self._load_provider(provider_type, api_key, model_name)
        
        # Initialize processors
        self.dedup_engine = DeduplicationEngine()
        self.transaction_processor = TransactionProcessor()
        self.metadata_extractor = MetadataExtractor(self.llm_provider)
        
        print(f"✅ Bank Statement Extractor initialized (v6.1)")
        print(f"   Provider: {provider_type}")
        print(f"   Model: {model_name}")
        print(f"   Mode: Page-by-page extraction (always)")
    
    def _load_provider(self, provider_type: str, api_key: str, model_name: str) -> LLMProvider:
        """Dynamically load the specified provider"""
        provider_config = config.PROVIDER_CONFIGS.get(provider_type)
        
        if not provider_config:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        # Import the provider module
        module = importlib.import_module(provider_config['module'])
        provider_class = getattr(module, provider_config['class'])
        
        # Instantiate and return
        return provider_class(api_key=api_key, model_name=model_name)
    
    def extract(self, pdf_path: str, output_dir: str = "output") -> Dict:
        """
        Extract transactions and metadata from bank statement
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for CSV and metadata
            
        Returns:
            Dict with extraction results
        """
        print(f"\n{'='*70}")
        print(f"Processing: {Path(pdf_path).name}")
        print(f"{'='*70}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read PDF
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        # Get page count
        page_count = self._get_page_count(pdf_path)
        print(f"📄 PDF has {page_count} page(s)")
        
        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(pdf_data)
        
        # Always use multi-page extraction (more reliable, avoids token limits)
        df = self._extract_multi_page(pdf_data, page_count)
        
        # Apply deduplication with transaction_id support
        df = self.dedup_engine.remove_duplicates(df)
        
        # Save results
        base_name = Path(pdf_path).stem
        transactions_path = os.path.join(output_dir, f"{base_name}_transactions.csv")
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        df.to_csv(transactions_path, index=False)
        print(f"💾 Transactions saved: {transactions_path}")
        
        if metadata:
            self.metadata_extractor.save_metadata(metadata, metadata_path)
        
        return {
            'transactions_file': transactions_path,
            'metadata_file': metadata_path if metadata else None,
            'transaction_count': len(df),
            'metadata': metadata
        }
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return len(pdf_reader.pages)
        except:
            return 1
    
    def _extract_multi_page(self, pdf_data: bytes, page_count: int) -> pd.DataFrame:
        """
        Process PDF page by page
        
        This method is ALWAYS used regardless of page count because:
        - More reliable (processes each page separately)
        - Avoids token limits (each page is a smaller request)
        - Better error handling (one page failing doesn't fail entire extraction)
        - More accurate (LLM focuses on one page at a time)
        """
        print(f"🤖 Page-by-page extraction ({page_count} page{'s' if page_count > 1 else ''})...")
        
        all_transactions = []
        
        for page_num in range(1, page_count + 1):
            print(f" 📄 Page {page_num}/{page_count}...", end=" ")
            
            prompt = SINGLE_PAGE_TRANSACTION_PROMPT.format(
                page_num=page_num,
                total_pages=page_count
            )
            
            try:
                response_text = self.llm_provider.extract_transactions(
                    pdf_data=pdf_data,
                    prompt=prompt,
                    temperature=config.TEMPERATURE,
                    max_tokens=10000
                )
                
                transactions = self.transaction_processor.parse_json(response_text)
                
                if transactions and len(transactions) > 0:
                    all_transactions.extend(transactions)
                    print(f"✓ {len(transactions)} txns")
                else:
                    print("⚠️ No data")
                    
            except Exception as e:
                print(f"❌ {e}")
        
        print(f"\n✓ Combined {len(all_transactions)} transactions from all pages")
        
        if not all_transactions:
            print("⚠️ No transactions extracted - empty statement or extraction failed")
            return pd.DataFrame(columns=['date', 'description', 'debit', 'credit', 'balance', 'transaction_id'])
        
        df = self.transaction_processor.create_dataframe(all_transactions)
        return df
    
    def _validate_totals(self, df: pd.DataFrame, metadata: dict) -> None:
        """Validate that totals match opening/closing balance"""
        if not metadata:
            return
            
        opening = metadata.get('opening_balance', 0)
        closing = metadata.get('closing_balance', 0)
        
        if opening == 0 and closing == 0:
            return  # Skip validation if metadata extraction failed
        
        total_credits = df['credit'].sum()
        total_debits = df['debit'].sum()
        
        calculated_closing = opening + total_credits - total_debits
        
        print(f"\n💰 Financial Validation:")
        print(f"   Opening Balance: ₹{opening:,.2f}")
        print(f"   Total Credits:   ₹{total_credits:,.2f}")
        print(f"   Total Debits:    ₹{total_debits:,.2f}")
        print(f"   Calculated Closing: ₹{calculated_closing:,.2f}")
        print(f"   Actual Closing:     ₹{closing:,.2f}")
        
        difference = abs(calculated_closing - closing)
        if difference < 10:
            print(f"   ✅ PERFECT MATCH (diff: ₹{difference:.2f})")
        else:
            print(f"   ⚠️ MISMATCH of ₹{difference:,.2f} - check for missing/extra transactions")