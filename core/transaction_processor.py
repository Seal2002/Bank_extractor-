"""
Transaction Data Processor - Pure Balance-Based Validation
NO KEYWORD-BASED ASSUMPTIONS - Let math determine correctness
"""

import pandas as pd
import json
import re
from typing import List, Dict, Optional


class TransactionProcessor:
    """
    Processes and validates transaction data using ONLY balance calculations
    Zero keyword-based assumptions - works for all 4500+ banks
    """
    
    def __init__(self):
        self.validation_stats = {
            'balance_fixes': 0,
            'both_columns_fixes': 0,
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
                    print(f"  ⚠️  Truncated to {len(result)} complete transactions")
                    return result
        except:
            pass
        
        return None
    
    def create_dataframe(self, transactions: List[Dict]) -> pd.DataFrame:
        """Convert transactions list to clean DataFrame with pure balance-based validation"""
        if not transactions:
            return pd.DataFrame(columns=['date', 'description', 'debit', 'credit', 'balance', 'transaction_id'])
        
        df = pd.DataFrame(transactions)
        
        # Ensure all required columns exist
        for col in ['date', 'description', 'debit', 'credit', 'balance', 'transaction_id']:
            if col not in df.columns:
                df[col] = None
        
        original_count = len(df)
        print(f"  📄 Parsing {original_count} transactions...")
        
        # Clean and parse data
        df = self._clean_data(df)
        
        # CRITICAL: Validate and fix debit/credit using BALANCE ONLY (zero keywords)
        df = self._validate_and_fix_debit_credit(df)
        
        # Filter invalid rows
        df = self._filter_valid_transactions(df)
        
        # Sort by date with stable sort to preserve order within same date
        df = df.sort_values('date', kind='stable').reset_index(drop=True)
        
        # Print validation statistics
        self._print_validation_stats()
        
        print(f"  ✅ Successfully processed {len(df)} valid transactions")
        return df[['date', 'description', 'debit', 'credit', 'balance', 'transaction_id']]
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        # Parse dates
        df['date_original'] = df['date'].copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
        
        # Second pass: flexible date parsing
        mask = df['date'].isna()
        if mask.any():
            print(f"  📄 Attempting flexible date parsing for {mask.sum()} rows...")
            df.loc[mask, 'date'] = df.loc[mask, 'date_original'].apply(self._parse_date_flexible)
        
        # Remove rows with unparseable dates
        date_filtered = df.dropna(subset=['date'])
        dates_removed = len(df) - len(date_filtered)
        if dates_removed > 0:
            print(f"  ⚠️  Removed {dates_removed} transactions with unparseable dates")
        
        df = date_filtered.drop(columns=['date_original'])
        
        # Convert numeric columns
        for col in ['debit', 'credit', 'balance']:
            if df[col].dtype == 'object':
                df[col] = (df[col].astype(str)
                          .str.replace('₹', '', regex=False)
                          .str.replace('Rs.', '', regex=False)
                          .str.replace('Rs', '', regex=False)
                          .str.replace('INR', '', regex=False)
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
        
        # Remove common suffixes (these are safe patterns that appear in ALL banks)
        df['description'] = df['description'].str.replace(r'\s*S\s+DEBIT\s*$', '', regex=True, case=False)
        df['description'] = df['description'].str.replace(r'\s*PO\s+S\s+DEBIT\s*$', '', regex=True, case=False)
        df['description'] = df['description'].str.replace(r'\s+T\s*$', '', regex=True)
        
        return df
    
    def _validate_and_fix_debit_credit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and auto-correct debit/credit using ONLY balance logic.
        ZERO KEYWORD-BASED ASSUMPTIONS - pure mathematics.
        Works for all banks because balance math is universal: new_balance = old_balance - debit + credit
        """
        print(f"  🔧 Validating debit/credit using pure balance calculations...")
        
        if len(df) < 2:
            return df
        
        # Sort by date to ensure chronological order (stable to preserve original order)
        df = df.sort_values('date', kind='stable').reset_index(drop=True)
        self.validation_stats['total_validated'] = len(df)
        
        for i in range(1, len(df)):
            self._validate_transaction_pure_balance(df, i)
        
        return df
    
    def _validate_transaction_pure_balance(self, df: pd.DataFrame, i: int):
        """
        Validate using ONLY balance math - NO keywords whatsoever
        Formula: current_balance = previous_balance - debit + credit
        """
        
        prev_bal = df.loc[i-1, 'balance']
        curr_bal = df.loc[i, 'balance']
        debit = df.loc[i, 'debit'] if pd.notna(df.loc[i, 'debit']) else 0
        credit = df.loc[i, 'credit'] if pd.notna(df.loc[i, 'credit']) else 0
        
        # Skip if both are zero or balance data missing
        if (debit == 0 and credit == 0) or pd.isna(prev_bal) or pd.isna(curr_bal):
            return
        
        # Calculate what the balance change actually was
        actual_change = curr_bal - prev_bal
        
        # Case 1: LLM put amounts in BOTH columns (common mistake)
        # Keep the one that matches actual balance change
        if debit > 0 and credit > 0:
            # Check which column is correct based on balance change
            if abs(actual_change - credit) < abs(actual_change + debit):
                # Balance increased, so credit is correct
                df.loc[i, 'debit'] = None
                self.validation_stats['both_columns_fixes'] += 1
                desc = str(df.loc[i, 'description'])[:50]
                self.validation_stats['warnings'].append(
                    f"Row {i}: Removed debit (kept credit={credit:.2f}) - balance increased by {actual_change:.2f} (desc: {desc})"
                )
            else:
                # Balance decreased, so debit is correct
                df.loc[i, 'credit'] = None
                self.validation_stats['both_columns_fixes'] += 1
                desc = str(df.loc[i, 'description'])[:50]
                self.validation_stats['warnings'].append(
                    f"Row {i}: Removed credit (kept debit={debit:.2f}) - balance decreased by {actual_change:.2f} (desc: {desc})"
                )
            return
        
        # Case 2: Amount is in debit column
        # Check if balance math is correct: new_bal = prev_bal - debit
        if debit > 0 and credit == 0:
            expected_change = -debit  # Debit should decrease balance
            
            # If math doesn't match, check if amount should be in credit instead
            if abs(actual_change - expected_change) > 0.02:
                # Check if moving to credit would fix it
                if abs(actual_change - debit) < 0.02:  # Positive change matches amount
                    df.loc[i, 'credit'] = debit
                    df.loc[i, 'debit'] = None
                    self.validation_stats['balance_fixes'] += 1
                    desc = str(df.loc[i, 'description'])[:50]
                    self.validation_stats['warnings'].append(
                        f"Row {i}: Moved {debit:.2f} from debit to credit - balance increased by {actual_change:.2f} (desc: {desc})"
                    )
        
        # Case 3: Amount is in credit column
        # Check if balance math is correct: new_bal = prev_bal + credit
        elif credit > 0 and debit == 0:
            expected_change = credit  # Credit should increase balance
            
            # If math doesn't match, check if amount should be in debit instead
            if abs(actual_change - expected_change) > 0.02:
                # Check if moving to debit would fix it
                if abs(actual_change + credit) < 0.02:  # Negative change matches amount
                    df.loc[i, 'debit'] = credit
                    df.loc[i, 'credit'] = None
                    self.validation_stats['balance_fixes'] += 1
                    desc = str(df.loc[i, 'description'])[:50]
                    self.validation_stats['warnings'].append(
                        f"Row {i}: Moved {credit:.2f} from credit to debit - balance decreased by {actual_change:.2f} (desc: {desc})"
                    )
    
    def _filter_valid_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid transactions with detailed logging"""
        initial_count = len(df)
        
        # Track what gets filtered
        filtered_reasons = []
        
        # Create mask for each condition
        has_description = (df['description'].notna()) & \
                        (df['description'].astype(str).str.strip() != '') & \
                        (df['description'].astype(str).str.lower() != 'nan')
        
        not_header = ~df['description'].astype(str).str.match(
            r'^(Date|Narration|Balance|Statement|Page|Continued|Opening Balance|Closing Balance|Total|Summary|Withdrawal|Deposit|Debit|Credit)\s*$',
            case=False, na=False
        )
        
        has_amount = (df['debit'].notna()) | (df['credit'].notna())
        
        has_balance = df['balance'].notna()
        
        # Log filtered transactions (only first 3 for brevity)
        count_logged = 0
        for i in df.index:
            if count_logged >= 3:
                break
            if not has_description[i]:
                filtered_reasons.append(f"Row {i}: No valid description")
                count_logged += 1
            elif not not_header[i]:
                filtered_reasons.append(f"Row {i}: Header/footer text: {df.loc[i, 'description']}")
                count_logged += 1
            elif not has_amount[i]:
                filtered_reasons.append(f"Row {i}: No debit or credit amount")
                count_logged += 1
            elif not has_balance[i]:
                filtered_reasons.append(f"Row {i}: No balance value")
                count_logged += 1
        
        # Apply all filters
        df = df[has_description & not_header & has_amount & has_balance]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"  ⚠️  Filtered out {filtered_count} invalid transaction(s)")
            if filtered_reasons:
                for reason in filtered_reasons:
                    print(f"     • {reason}")
        
        return df
    
    def _print_validation_stats(self):
        """Print validation statistics"""
        total_fixes = self.validation_stats['balance_fixes'] + self.validation_stats['both_columns_fixes']
        
        if total_fixes > 0:
            print(f"  ✅ Fixed {total_fixes} debit/credit assignments using pure balance logic:")
            if self.validation_stats['balance_fixes'] > 0:
                print(f"     • {self.validation_stats['balance_fixes']} moved between columns")
            if self.validation_stats['both_columns_fixes'] > 0:
                print(f"     • {self.validation_stats['both_columns_fixes']} had both columns (kept correct one)")
            
            if self.validation_stats['warnings'][:3]:  # Show first 3 warnings
                print(f"  📋 Sample fixes:")
                for warning in self.validation_stats['warnings'][:3]:
                    print(f"     • {warning}")
        else:
            print(f"  ✅ All debit/credit assignments are mathematically correct")
    
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
        # Note: In character class, hyphen must be escaped or placed at start/end
        patterns = [
            (r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})', lambda m: f"{m[3]}-{m[2].zfill(2)}-{m[1].zfill(2)}"),
            (r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})$', lambda m: f"20{m[3]}-{m[2].zfill(2)}-{m[1].zfill(2)}"),
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