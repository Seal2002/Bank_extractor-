"""
Main script to run the bank statement extractor
"""
import os
import sys
# Change this import:
from core.bank_statement_extractor import BankStatementExtractor


def main():
    # Example 1: Use default provider from config (Gemini)
    extractor = BankStatementExtractor()
    
    # Example 2: Override provider (when you want to switch)
    # extractor = BankStatementExtractor(
    #     provider_type="openai",
    #     api_key="your-openai-key",
    #     model_name="gpt-4-vision-preview"
    # )
    
    # Process a single file
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = extractor.extract(pdf_path, output_dir="extracted_data")
        
        print(f"\n{'='*70}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Transactions: {result['transaction_count']}")
        print(f"Transactions file: {result['transactions_file']}")
        print(f"Metadata file: {result['metadata_file']}")
    else:
        print("Usage: python main.py <path_to_pdf>")
        print("Example: python main.py bank_statement.pdf")

if __name__ == "__main__":
    main()
