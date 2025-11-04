"""
Transaction Deduplication Engine
SIMPLE RULE: Remove duplicates ONLY if ALL fields match exactly
"""
import pandas as pd

class DeduplicationEngine:
    """Handles transaction deduplication with strict matching"""
    
    def __init__(self):
        pass
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates ONLY if ALL 6 fields match exactly:
        - date, description, debit, credit, balance, transaction_id
        
        If transaction_id is different → NOT a duplicate (keep both)
        If ALL fields identical → duplicate (keep first, remove rest)
        """
        if len(df) == 0:
            return df
        
        original_count = len(df)
        print(f"\n🔍 Checking for duplicates in {original_count} transactions...")
        
        # Ensure transaction_id column exists
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = None
        
        # Remove EXACT duplicates only (all 6 fields must match)
        df_dedup = df.drop_duplicates(
            subset=['date', 'description', 'debit', 'credit', 'balance', 'transaction_id'],
            keep='first'  # Keep first occurrence, remove rest
        )
        
        total_removed = original_count - len(df_dedup)
        
        if total_removed > 0:
            print(f" ✅ Removed {total_removed} exact duplicates (all fields matched)")
            print(f" 📊 Final count: {len(df_dedup)} unique transactions")
        else:
            print(f" ✅ No duplicates found")
        
        return df_dedup.reset_index(drop=True)
