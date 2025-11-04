"""
Comprehensive Test Suite for Phase 2 - Multi-Entity Transaction Analysis
Tests the system with multiple bank statements from different banks
"""
import os
import sys
import pandas as pd
from pathlib import Path
from core.bank_statement_extractor import BankStatementExtractor
from core.transaction_matcher import TransactionMatcher


def test_extraction(pdf_path: str, entity_name: str):
    """Test extraction on a single PDF"""
    print(f"\n{'='*80}")
    print(f"Testing: {entity_name}")
    print(f"File: {pdf_path}")
    print(f"{'='*80}")
    
    try:
        extractor = BankStatementExtractor()
        result = extractor.extract(pdf_path, output_dir="test_results")
        
        print(f"✅ Extraction successful!")
        print(f"   Transactions: {result['transaction_count']}")
        print(f"   Output: {result['transactions_file']}")
        
        # Load and display sample
        df = pd.read_csv(result['transactions_file'])
        print(f"\n📊 Sample transactions:")
        print(df.head(3).to_string())
        
        return {
            'success': True,
            'df': df,
            'metadata': result.get('metadata'),
            'entity_name': entity_name
        }
    
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return {'success': False, 'error': str(e)}


def test_matching(entity1_data: dict, entity2_data: dict):
    """Test transaction matching between two entities"""
    print(f"\n{'='*80}")
    print(f"Testing Transaction Matching")
    print(f"{'='*80}")
    
    try:
        matcher = TransactionMatcher()
        
        matches = matcher.match_transactions(
            entity1_data['df'],
            entity2_data['df'],
            entity1_data['entity_name'],
            entity2_data['entity_name']
        )
        
        if len(matches) == 0:
            print("⚠️  No matches found between these entities")
            return None
        
        # Analyze matches
        analysis = matcher.analyze_matches(matches)
        
        # Display results
        print(f"\n📊 Match Analysis:")
        print(f"   Total Matches: {analysis['total_transactions']}")
        print(f"   Total Value: ₹{analysis['total_value']:,.2f}")
        print(f"   High-Value (>10K): {analysis['high_value_transactions']} (₹{analysis['high_value_total']:,.2f})")
        print(f"   Date Range: {analysis['first_transaction_date']} to {analysis['last_transaction_date']}")
        
        # Save results
        os.makedirs("test_results", exist_ok=True)
        matches_file = f"test_results/matches_{entity1_data['entity_name']}_{entity2_data['entity_name']}.csv"
        matches.to_csv(matches_file, index=False)
        print(f"\n💾 Matches saved to: {matches_file}")
        
        # Display sample matches
        print(f"\n📋 Sample matched transactions:")
        print(matches.head(5)[['transaction_date', 'amount', f"{entity1_data['entity_name']}_action"]].to_string())
        
        # High-value transactions
        high_value = matcher.get_high_value_transactions(matches)
        if len(high_value) > 0:
            print(f"\n🚨 High-Value Transactions (>₹10,000):")
            print(high_value[['transaction_date', 'amount']].to_string(index=False))
            
            high_value_file = f"test_results/high_value_{entity1_data['entity_name']}_{entity2_data['entity_name']}.csv"
            high_value.to_csv(high_value_file, index=False)
            print(f"\n💾 High-value transactions saved to: {high_value_file}")
        
        # Generate summary
        summary = matcher.generate_transaction_summary(
            matches, 
            entity1_data['entity_name'], 
            entity2_data['entity_name']
        )
        
        summary_file = f"test_results/summary_{entity1_data['entity_name']}_{entity2_data['entity_name']}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"\n💾 Summary report saved to: {summary_file}")
        
        return matches
    
    except Exception as e:
        print(f"❌ Matching failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_tests():
    """Run tests on all available bank statements"""
    print("="*80)
    print("COMPREHENSIVE TEST SUITE - PHASE 2")
    print("Multi-Entity Transaction Analysis")
    print("="*80)
    
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Get all PDFs from input directory
    input_dir = Path("input")
    if not input_dir.exists():
        print("❌ Input directory not found. Please create 'input' folder and add PDF bank statements.")
        return
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if len(pdf_files) < 2:
        print(f"❌ Need at least 2 PDF files for testing. Found {len(pdf_files)}.")
        print("   Please add more bank statements to the 'input' folder.")
        return
    
    print(f"\n📁 Found {len(pdf_files)} PDF files in input directory")
    
    # Test extraction on all files
    extracted_data = []
    for pdf_file in pdf_files:
        entity_name = pdf_file.stem.replace('_', ' ')
        result = test_extraction(str(pdf_file), entity_name)
        
        if result['success']:
            extracted_data.append(result)
    
    print(f"\n✅ Successfully extracted {len(extracted_data)} statements")
    
    if len(extracted_data) < 2:
        print("❌ Need at least 2 successful extractions to test matching.")
        return
    
    # Test matching between all pairs
    print(f"\n{'='*80}")
    print("Testing All Entity Pairs")
    print(f"{'='*80}")
    
    match_results = []
    for i in range(len(extracted_data)):
        for j in range(i + 1, len(extracted_data)):
            entity1 = extracted_data[i]
            entity2 = extracted_data[j]
            
            print(f"\n🔄 Testing: {entity1['entity_name']} ↔ {entity2['entity_name']}")
            
            matches = test_matching(entity1, entity2)
            if matches is not None:
                match_results.append({
                    'entity1': entity1['entity_name'],
                    'entity2': entity2['entity_name'],
                    'match_count': len(matches)
                })
    
    # Summary report
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total PDFs tested: {len(pdf_files)}")
    print(f"Successful extractions: {len(extracted_data)}")
    print(f"Entity pairs tested: {len(match_results)}")
    
    if match_results:
        print(f"\n📊 Matching Results:")
        for result in match_results:
            print(f"   {result['entity1']} ↔ {result['entity2']}: {result['match_count']} matches")
    
    print(f"\n💾 All test results saved to: test_results/")
    print(f"{'='*80}")


def test_csv_upload():
    """Test CSV upload functionality"""
    print(f"\n{'='*80}")
    print("Testing CSV Upload Functionality")
    print(f"{'='*80}")
    
    # Create sample CSV data
    sample_data = {
        'date': ['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15'],
        'description': ['NEFT Transfer', 'Salary Credit', 'ATM Withdrawal', 'UPI Payment'],
        'debit': [5000.00, None, 2000.00, 500.00],
        'credit': [None, 50000.00, None, None],
        'balance': [45000.00, 95000.00, 93000.00, 92500.00],
        'transaction_id': ['NEFT123', 'SAL001', None, 'UPI456']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample CSV
    csv_file = "test_results/sample_entity.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"✅ Created sample CSV: {csv_file}")
    print(f"\n📊 Sample data:")
    print(df.to_string())
    
    return csv_file


if __name__ == "__main__":
    print("🚀 Starting Phase 2 Test Suite\n")
    
    # Run comprehensive tests
    run_comprehensive_tests()
    
    # Test CSV upload
    print("\n")
    test_csv_upload()
    
    print("\n✅ All tests completed!")
    print("\n📌 Next steps:")
    print("   1. Review test results in 'test_results' folder")
    print("   2. Run dashboard: streamlit run dashboard_multi_entity.py")
    print("   3. Upload statements and test matching in the web interface")
