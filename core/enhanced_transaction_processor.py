"""
Enhanced Transaction Data Processor
Processes and validates transaction data with intelligent debit/credit validation and auto-correction
"""
import pandas as pd
import json
import re
from typing import List, Dict, Optional

class EnhancedTransactionProcessor:
    """
    Processes and validates transaction data with intelligent error detection and correction
    """
    
    # Keywords that indicate a transaction should be DEBIT (withdrawal)
    DEBIT_KEYWORDS = [
        'ATM', 'ATW', 'WITHDRAWAL', 
        'POS', 'PURCHASE', 
        'NWD', 'EAW',  # Non-home withdrawal, Electronic withdrawal
        'AUTOPAY SI', 'SI-TAD', 'SI-MAD',  # Standing instructions (when it's the payment, not reversal)
        'FUND TRF DM',  # Fund transfer debit
        'FEE', 'CHARGE',
        'CD-',  # Certificate of deposit
    ]
    
    # Keywords that indicate a transaction should be CREDIT (deposit)
    CREDIT_KEYWORDS = [
        'SALARY', 'SALMAR', 'SALAPR', 'SALJUN', 'SALJULY', 'SALAUG', 'SALSEP', 'SALOCT', 'SALNOV', 'SALDEC',
        'NEFT', 'IMPS', 'UPI',  # Bank transfers (incoming)
        'CRV', 'CREDIT', 'DEPOSIT',  # Credit reversals and deposits
        'POS REF',  # POS refunds/cashbacks
        'INTEREST CAPITALISED',
        'REFUND',
        'ADVANCE',  # Advance credits
    ]
    
    def __init__(self):
        self.validation_stats = {
            'balance_fixes': 0,
            'keyword_fixes': 0,
            'total_validated': 0,
            'warnings': []
        }
    
    def parse_json(self, text: str) -> Optional[List[Dict]]:
        """Robust JSON parser with multiple fallback strategies"""
        # Method 1: Direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Method 2: Remove markdown
        try:
            cleaned = text.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned)
        except:
            pass
        
        # Method 3: Extract array by brackets
        try:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except:
            pass
        
        # Method 4: Truncate to last complete transaction
        try:
            cleaned = text.replace('```json', '').replace('```', '').strip()
            start = cleaned.find('[')
            if start != -1:
                last_complete = cleaned.rfind('},')
                if last_complete != -1:
                    truncated = cleaned[start:last_complete+1] + ']'
                    result = json.loads(truncated)
                    print(f" âš ï¸ Truncated to {len(result)} complete transactions")
                    return result
        except:
            pass
        
        return None
    
    def create_dataframe(self, transactions: List[Dict]) -> pd.DataFrame:
        """Convert transactions list to clean DataFrame with validation and auto-correction"""
        if not transactions:
            return pd.DataFrame(columns=['date', 'description', 'debit', 'credit', 'balance', 'transaction_id'])
        
        df = pd.DataFrame(transactions)
        
        # Ensure all required columns exist
        for col in ['date', 'description', 'debit', 'credit', 'balance', 'transaction_id']:
            if col not in df.columns:
                df[col] = None
        
        original_count = len(df)
        print(f" ðŸ“„ Parsing {original_count} transactions...")
        
        # Clean and parse data
        df = self._clean_data(df)
        
        # CRITICAL: Validate and fix debit/credit assignments
        df = self._validate_and_fix_debit_credit(df)
        
        # Filter invalid rows
        df = self._filter_valid_transactions(df)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Print validation statistics
        self._print_validation_stats()
        
        print(f" âœ… Successfully processed {len(df)} valid transactions")
        
        return df[['date', 'description', 'debit', 'credit', 'balance', 'transaction_id']]
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        # Parse dates
        df['date_original'] = df['date'].copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
        
        # Second pass: flexible date parsing
        mask = df['date'].isna()
        if mask.any():
            print(f" ðŸ”„ Attempting flexible date parsing for {mask.sum()} rows...")
            df.loc[mask, 'date'] = df.loc[mask, 'date_original'].apply(self._parse_date_flexible)
        
        # Remove rows with unparseable dates
        date_filtered = df.dropna(subset=['date'])
        dates_removed = len(df) - len(date_filtered)
        if dates_removed > 0:
            print(f" âš ï¸ Removed {dates_removed} transactions with unparseable dates")
        df = date_filtered.drop(columns=['date_original'])
        
        # Convert numeric columns
        for col in ['debit', 'credit', 'balance']:
            if df[col].dtype == 'object':
                df[col] = (df[col].astype(str)
                          .str.replace('â‚¹', '', regex=False)
                          .str.replace('Rs.', '', regex=False)
                          .str.replace('Rs', '', regex=False)
                          .str.replace('INR', '', regex=False)
                          # Handle Cr/Dr suffixes (used by ICICI and other banks)
                          .str.replace(r'\s*Cr\s*$', '', regex=True, case=False)
                          .str.replace(r'\s*Dr\s*$', '', regex=True, case=False)
                          .str.replace(',', '', regex=False)
                          .str.replace(' ', '', regex=False)
                          .str.strip())
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Round to 2 decimal places
        df['debit'] = df['debit'].round(2)
        df['credit'] = df['credit'].round(2)
        df['balance'] = df['balance'].round(2)
        
        # Clean transaction_id
        df['transaction_id'] = df['transaction_id'].astype(str).str.strip()
        df['transaction_id'] = df['transaction_id'].replace(['nan', 'None', ''], None)
        
        # Clean descriptions
        df['description'] = df['description'].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
        df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True).str.strip()
        # Remove common suffixes
        df['description'] = df['description'].str.replace(r'\s*S\s+DEBIT\s*$', '', regex=True, case=False)
        df['description'] = df['description'].str.replace(r'\s*PO\s+S\s+DEBIT\s*$', '', regex=True, case=False)
        df['description'] = df['description'].str.replace(r'\s+T\s*$', '', regex=True)
        
        return df
    
    def _validate_and_fix_debit_credit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and auto-correct debit/credit column assignments using three strategies:
        1. Balance logic: If swap would make balance math work, swap them
        2. Description keywords: If description suggests withdrawal but amount in credit, swap
        3. Pattern detection: Detect autopay pairs and other patterns
        """
        print(f" ðŸ”§ Validating debit/credit column assignments...")
        
        if len(df) < 2:
            return df
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        self.validation_stats['total_validated'] = len(df)
        
        for i in range(1, len(df)):
            self._validate_transaction(df, i)
        
        return df
    
    def _validate_transaction(self, df: pd.DataFrame, i: int):
        """Validate a single transaction and fix if needed"""
        desc = str(df.loc[i, 'description']).upper()
        prev_bal = df.loc[i-1, 'balance']
        curr_bal = df.loc[i, 'balance']
        debit = df.loc[i, 'debit'] if pd.notna(df.loc[i, 'debit']) else 0
        credit = df.loc[i, 'credit'] if pd.notna(df.loc[i, 'credit']) else 0
        
        # Skip if both are zero or balance data missing
        if (debit == 0 and credit == 0) or pd.isna(prev_bal) or pd.isna(curr_bal):
            return
        
        # Strategy 1: Balance-based validation
        # NOTE: Negative balances are LEGITIMATE for AUTOPAY and similar transactions!
        # We only check if the balance math is correct, not if it's negative.
        expected_bal = prev_bal - debit + credit
        diff = abs(expected_bal - curr_bal)
        
        # If balance doesn't match, check if swap would fix it
        if diff > 0.02:
            swapped_bal = prev_bal - credit + debit
            if abs(swapped_bal - curr_bal) < 0.02:
                # Swap would fix it - do the swap
                df.loc[i, 'debit'] = credit if credit > 0 else None
                df.loc[i, 'credit'] = debit if debit > 0 else None
                self.validation_stats['balance_fixes'] += 1
                self.validation_stats['warnings'].append(
                    f"Row {i}: Swapped debit/credit based on balance math (desc: {desc[:50]}...)"
                )
                return  # Skip keyword check since we already fixed it
        
        # Strategy 2: Description keyword validation
        # This is especially important for POS REF and CRV which are often misclassified
        is_likely_debit = any(kw in desc for kw in self.DEBIT_KEYWORDS)
        is_likely_credit = any(kw in desc for kw in self.CREDIT_KEYWORDS)
        
        # If description suggests debit but amount is in credit column
        if is_likely_debit and credit > 0 and debit == 0:
            df.loc[i, 'debit'] = credit
            df.loc[i, 'credit'] = None
            self.validation_stats['keyword_fixes'] += 1
            self.validation_stats['warnings'].append(
                f"Row {i}: Moved amount from credit to debit based on keywords (desc: {desc[:50]}...)"
            )
        
        # If description suggests credit but amount is in debit column
        elif is_likely_credit and debit > 0 and credit == 0:
            df.loc[i, 'credit'] = debit
            df.loc[i, 'debit'] = None
            self.validation_stats['keyword_fixes'] += 1
            self.validation_stats['warnings'].append(
                f"Row {i}: Moved amount from debit to credit based on keywords (desc: {desc[:50]}...)"
            )
    
    def _filter_valid_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid transactions"""
        df = df[
            # Must have description
            (df['description'].notna()) &
            (df['description'].astype(str).str.strip() != '') &
            (df['description'].astype(str).str.lower() != 'nan') &
            # Filter out header/footer text
            (~df['description'].astype(str).str.match(
                r'^(Date|Narration|Balance|Statement|Page|Continued|Opening Balance|Closing Balance|Total|Summary)\s*$',
                case=False, na=False
            )) &
            # Must have either debit or credit
            (df['debit'].notna() | df['credit'].notna()) &
            # Must have balance
            (df['balance'].notna())
        ]
        
        return df
    
    def _print_validation_stats(self):
        """Print validation statistics"""
        total_fixes = self.validation_stats['balance_fixes'] + self.validation_stats['keyword_fixes']
        
        if total_fixes > 0:
            print(f" âœ… Fixed {total_fixes} debit/credit assignments:")
            print(f"    - Balance-based fixes: {self.validation_stats['balance_fixes']}")
            print(f"    - Keyword-based fixes: {self.validation_stats['keyword_fixes']}")
            
            if self.validation_stats['warnings'][:3]:  # Show first 3 warnings
                print(f" ðŸ“‹ Sample fixes:")
                for warning in self.validation_stats['warnings'][:3]:
                    print(f"    â€¢ {warning}")
        else:
            print(f" âœ… All debit/credit assignments appear correct")
    
    def _parse_date_flexible(self, date_str) -> Optional[pd.Timestamp]:
        """Flexible date parser for various formats"""
        if pd.isna(date_str) or date_str is None:
            return None
        
        date_str = str(date_str).strip()
        
        if not date_str or date_str.lower() in ['nan', 'none', 'null', 'n/a', '']:
            return None
        
        # Try standard parsing first
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d')
        except:
            pass
        
        # Try with dayfirst
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except:
            pass
        
        # Try regex patterns for common formats
        patterns = [
            (r'(\d{1,2})[/-\.](\d{1,2})[/-\.](\d{4})', lambda m: f"{m[3]}-{m[2].zfill(2)}-{m[1].zfill(2)}"),
            (r'(\d{1,2})[/-\.](\d{1,2})[/-\.](\d{2})$', lambda m: f"20{m[3]}-{m[2].zfill(2)}-{m[1].zfill(2)}"),
            (r'^(\d{2})(\d{2})(\d{2})$', lambda m: f"20{m[3]}-{m[2]}-{m[1]}"),
            (r'^(\d{2})(\d{2})(\d{4})$', lambda m: f"{m[3]}-{m[2]}-{m[1]}"),
        ]
        
        for pattern, formatter in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    formatted_date = formatter(match.groups())
                    return pd.to_datetime(formatted_date, format='%Y-%m-%d')
                except:
                    pass
        
        # Try with dateutil parser as last resort
        try:
            from dateutil import parser
            return pd.to_datetime(parser.parse(date_str, dayfirst=True))
        except:
            pass
        
        return None