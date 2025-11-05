"""
Pattern Detection Engine
Identifies suspicious transaction patterns like chunking, rapid withdrawals, etc.
"""

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PatternDetector:
    """
    Detects suspicious transaction patterns in matched transactions:
    1. Chunking: Multiple small transactions that sum to a large amount
    2. Rapid withdrawal: Money received and withdrawn within 72 hours
    3. Round amount patterns: Suspiciously round transactions
    """
    
    def __init__(
        self,
        chunking_window_days: int = 30,
        chunking_similarity_threshold: float = 0.10,  # 10% variance allowed
        rapid_withdrawal_hours: int = 72,
        round_amount_threshold: int = 10000,
        min_chunk_count: int = 3
    ):
        """
        Initialize pattern detector with configurable parameters
        
        Args:
            chunking_window_days: Time window to look for chunking patterns (default: 30 days)
            chunking_similarity_threshold: How similar amounts should be to be considered chunks (10% variance)
            rapid_withdrawal_hours: Time window for rapid withdrawal detection (default: 72 hours)
            round_amount_threshold: Minimum amount to check for round number patterns
            min_chunk_count: Minimum number of similar transactions to flag as chunking
        """
        self.chunking_window_days = chunking_window_days
        self.chunking_similarity_threshold = chunking_similarity_threshold
        self.rapid_withdrawal_hours = rapid_withdrawal_hours
        self.round_amount_threshold = round_amount_threshold
        self.min_chunk_count = min_chunk_count
        
        self.detected_patterns = {
            'chunking_patterns': [],
            'rapid_withdrawals': [],
            'round_amount_patterns': [],
            'statistics': {}
        }
    
    def analyze_patterns(
        self,
        matches_df: pd.DataFrame,
        entity1_df: pd.DataFrame,
        entity2_df: pd.DataFrame,
        entity1_name: str,
        entity2_name: str
    ) -> Dict:
        """
        Analyze all patterns in the matched transactions
        
        Args:
            matches_df: DataFrame with matched transactions
            entity1_df: Entity 1's full transaction history
            entity2_df: Entity 2's full transaction history
            entity1_name: Name of Entity 1
            entity2_name: Name of Entity 2
            
        Returns:
            Dictionary with all detected patterns
        """
        print(f"\n🔍 Analyzing transaction patterns...")
        
        if len(matches_df) == 0:
            print("   ⚠️  No matches to analyze")
            return self.detected_patterns
        
        # Convert dates
        matches_df = matches_df.copy()
        matches_df['transaction_date'] = pd.to_datetime(matches_df['transaction_date'])
        entity1_df = entity1_df.copy()
        entity1_df['date'] = pd.to_datetime(entity1_df['date'])
        entity2_df = entity2_df.copy()
        entity2_df['date'] = pd.to_datetime(entity2_df['date'])
        
        # 1. Detect chunking patterns
        chunking_patterns = self._detect_chunking_patterns(matches_df, entity1_name, entity2_name)
        self.detected_patterns['chunking_patterns'] = chunking_patterns
        
        # 2. Detect rapid withdrawals
        rapid_withdrawals = self._detect_rapid_withdrawals(
            matches_df, entity2_df, entity1_name, entity2_name
        )
        self.detected_patterns['rapid_withdrawals'] = rapid_withdrawals
        
        # 3. Detect round amount patterns
        round_patterns = self._detect_round_amount_patterns(matches_df)
        self.detected_patterns['round_amount_patterns'] = round_patterns
        
        # Calculate statistics
        self._calculate_statistics(matches_df)
        
        # Print summary
        self._print_pattern_summary()
        
        return self.detected_patterns
    
    def _detect_chunking_patterns(
        self,
        matches_df: pd.DataFrame,
        entity1_name: str,
        entity2_name: str
    ) -> List[Dict]:
        """
        Detect chunking patterns: Multiple similar transactions in a time window
        Example: 10 transactions of ₹1,00,000 each totaling ₹10,00,000
        """
        print(f"   🔍 Checking for chunking patterns...")
        
        chunking_patterns = []
        
        # Sort by date
        df = matches_df.sort_values('transaction_date').reset_index(drop=True)
        
        # Group by direction (who sent to whom)
        for action_col in [f'{entity1_name}_action', f'{entity2_name}_action']:
            if action_col not in df.columns:
                continue
            
            # Get transactions where one entity sent money
            sent_df = df[df[action_col] == 'sent'].copy()
            
            if len(sent_df) < self.min_chunk_count:
                continue
            
            # Look for groups of similar amounts within time window
            for i in range(len(sent_df)):
                current_date = sent_df.iloc[i]['transaction_date']
                current_amount = sent_df.iloc[i]['amount']
                
                # Define window
                window_start = current_date
                window_end = current_date + timedelta(days=self.chunking_window_days)
                
                # Find transactions in window
                window_txns = sent_df[
                    (sent_df['transaction_date'] >= window_start) &
                    (sent_df['transaction_date'] <= window_end)
                ].copy()
                
                if len(window_txns) < self.min_chunk_count:
                    continue
                
                # Check if amounts are similar (within threshold)
                amounts = window_txns['amount'].values
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)
                cv = std_amount / mean_amount if mean_amount > 0 else 0
                
                # If coefficient of variation is low, amounts are similar
                if cv <= self.chunking_similarity_threshold and len(window_txns) >= self.min_chunk_count:
                    # Check if we already flagged this pattern
                    pattern_key = f"{window_start.date()}_{mean_amount:.0f}_{len(window_txns)}"
                    if not any(p.get('pattern_key') == pattern_key for p in chunking_patterns):
                        sender = entity1_name if action_col == f'{entity1_name}_action' else entity2_name
                        receiver = entity2_name if action_col == f'{entity1_name}_action' else entity1_name
                        
                        pattern = {
                            'pattern_key': pattern_key,
                            'pattern_type': 'chunking',
                            'severity': 'high' if len(window_txns) >= 5 else 'medium',
                            'sender': sender,
                            'receiver': receiver,
                            'transaction_count': len(window_txns),
                            'average_amount': float(mean_amount),
                            'total_amount': float(window_txns['amount'].sum()),
                            'first_transaction_date': window_txns['transaction_date'].min().strftime('%Y-%m-%d'),
                            'last_transaction_date': window_txns['transaction_date'].max().strftime('%Y-%m-%d'),
                            'days_span': (window_txns['transaction_date'].max() - window_txns['transaction_date'].min()).days,
                            'amount_variance': float(cv),
                            'description': f"{sender} sent {len(window_txns)} similar transactions averaging ₹{mean_amount:,.0f} (total ₹{window_txns['amount'].sum():,.0f}) to {receiver} over {(window_txns['transaction_date'].max() - window_txns['transaction_date'].min()).days} days"
                        }
                        chunking_patterns.append(pattern)
        
        print(f"   {'✅' if len(chunking_patterns) == 0 else '⚠️ '} Found {len(chunking_patterns)} chunking pattern(s)")
        return chunking_patterns
    
    def _detect_rapid_withdrawals(
        self,
        matches_df: pd.DataFrame,
        entity2_df: pd.DataFrame,
        entity1_name: str,
        entity2_name: str
    ) -> List[Dict]:
        """
        Detect rapid withdrawals: Entity 2 receives money and withdraws it within 72 hours
        """
        print(f"   🔍 Checking for rapid withdrawals...")
        
        rapid_withdrawals = []
        
        # Get transactions where Entity 2 received money
        received_col = f'{entity2_name}_action'
        if received_col not in matches_df.columns:
            return rapid_withdrawals
        
        received_df = matches_df[matches_df[received_col] == 'received'].copy()
        
        for _, received_txn in received_df.iterrows():
            received_date = pd.to_datetime(received_txn['transaction_date'])
            received_amount = received_txn['amount']
            
            # Look for withdrawals in Entity 2's transactions within 72 hours
            window_end = received_date + timedelta(hours=self.rapid_withdrawal_hours)
            
            # Find withdrawals (debits) in Entity 2's account
            withdrawals = entity2_df[
                (entity2_df['date'] > received_date) &
                (entity2_df['date'] <= window_end) &
                (entity2_df['debit'].notna()) &
                (entity2_df['debit'] > 0)
            ].copy()
            
            if len(withdrawals) > 0:
                # Check if total withdrawals are close to received amount
                total_withdrawn = withdrawals['debit'].sum()
                
                # If withdrawn amount is >= 80% of received amount, flag it
                if total_withdrawn >= received_amount * 0.8:
                    hours_diff = (withdrawals['date'].min() - received_date).total_seconds() / 3600
                    
                    pattern = {
                        'pattern_type': 'rapid_withdrawal',
                        'severity': 'high' if hours_diff <= 24 else 'medium',
                        'entity': entity2_name,
                        'received_amount': float(received_amount),
                        'withdrawn_amount': float(total_withdrawn),
                        'withdrawal_percentage': float(total_withdrawn / received_amount * 100),
                        'received_date': received_date.strftime('%Y-%m-%d %H:%M'),
                        'first_withdrawal_date': withdrawals['date'].min().strftime('%Y-%m-%d %H:%M'),
                        'hours_difference': float(hours_diff),
                        'withdrawal_count': len(withdrawals),
                        'description': f"{entity2_name} received ₹{received_amount:,.0f} and withdrew ₹{total_withdrawn:,.0f} ({total_withdrawn/received_amount*100:.0f}%) within {hours_diff:.0f} hours ({len(withdrawals)} withdrawals)"
                    }
                    rapid_withdrawals.append(pattern)
        
        print(f"   {'✅' if len(rapid_withdrawals) == 0 else '⚠️ '} Found {len(rapid_withdrawals)} rapid withdrawal pattern(s)")
        return rapid_withdrawals
    
    def _detect_round_amount_patterns(self, matches_df: pd.DataFrame) -> List[Dict]:
        """
        Detect suspiciously round amounts (e.g., exactly ₹1,00,000, ₹5,00,000)
        """
        print(f"   🔍 Checking for round amount patterns...")
        
        round_patterns = []
        
        # Check for round amounts above threshold
        high_value_df = matches_df[matches_df['amount'] >= self.round_amount_threshold].copy()
        
        for _, txn in high_value_df.iterrows():
            amount = txn['amount']
            
            # Check if amount is suspiciously round (divisible by 10000, 50000, 100000, etc.)
            is_round = False
            round_level = None
            
            if amount % 100000 == 0:
                is_round = True
                round_level = '1,00,000'
            elif amount % 50000 == 0:
                is_round = True
                round_level = '50,000'
            elif amount % 10000 == 0:
                is_round = True
                round_level = '10,000'
            
            if is_round:
                pattern = {
                    'pattern_type': 'round_amount',
                    'severity': 'low',
                    'amount': float(amount),
                    'round_level': round_level,
                    'transaction_date': txn['transaction_date'],
                    'description': f"Round amount: ₹{amount:,.0f} (multiple of ₹{round_level})"
                }
                round_patterns.append(pattern)
        
        print(f"   {'✅' if len(round_patterns) == 0 else 'ℹ️ '} Found {len(round_patterns)} round amount pattern(s)")
        return round_patterns
    
    def _calculate_statistics(self, matches_df: pd.DataFrame):
        """Calculate overall pattern statistics"""
        self.detected_patterns['statistics'] = {
            'total_transactions_analyzed': len(matches_df),
            'total_patterns_detected': (
                len(self.detected_patterns['chunking_patterns']) +
                len(self.detected_patterns['rapid_withdrawals']) +
                len(self.detected_patterns['round_amount_patterns'])
            ),
            'high_severity_patterns': sum(
                1 for p in self.detected_patterns['chunking_patterns'] + 
                self.detected_patterns['rapid_withdrawals']
                if p.get('severity') == 'high'
            ),
            'chunking_patterns_detected': len(self.detected_patterns['chunking_patterns']),
            'rapid_withdrawal_patterns_detected': len(self.detected_patterns['rapid_withdrawals']),
            'round_amount_patterns_detected': len(self.detected_patterns['round_amount_patterns'])
        }
    
    def _print_pattern_summary(self):
        """Print summary of detected patterns"""
        stats = self.detected_patterns['statistics']
        
        print(f"\n📊 Pattern Detection Summary:")
        print(f"   Total Patterns Detected: {stats['total_patterns_detected']}")
        print(f"   🚨 High Severity: {stats['high_severity_patterns']}")
        print(f"   🔸 Chunking: {stats['chunking_patterns_detected']}")
        print(f"   💨 Rapid Withdrawals: {stats['rapid_withdrawal_patterns_detected']}")
        print(f"   ⭕ Round Amounts: {stats['round_amount_patterns_detected']}")
        
        # Print details of high severity patterns
        if stats['high_severity_patterns'] > 0:
            print(f"\n⚠️  High Severity Patterns:")
            for pattern in self.detected_patterns['chunking_patterns']:
                if pattern.get('severity') == 'high':
                    print(f"   🔸 {pattern['description']}")
            for pattern in self.detected_patterns['rapid_withdrawals']:
                if pattern.get('severity') == 'high':
                    print(f"   💨 {pattern['description']}")
    
    def generate_pattern_report(self) -> str:
        """Generate a detailed text report of all detected patterns"""
        report = []
        report.append("="*80)
        report.append("TRANSACTION PATTERN ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        stats = self.detected_patterns['statistics']
        report.append(f"Analysis Summary:")
        report.append(f"- Total Transactions Analyzed: {stats['total_transactions_analyzed']}")
        report.append(f"- Total Patterns Detected: {stats['total_patterns_detected']}")
        report.append(f"- High Severity Patterns: {stats['high_severity_patterns']}")
        report.append("")
        
        # Chunking patterns
        if self.detected_patterns['chunking_patterns']:
            report.append("="*80)
            report.append("🔸 CHUNKING PATTERNS (Multiple Similar Transactions)")
            report.append("="*80)
            for i, pattern in enumerate(self.detected_patterns['chunking_patterns'], 1):
                report.append(f"\nPattern #{i} [{pattern['severity'].upper()} SEVERITY]:")
                report.append(f"  Description: {pattern['description']}")
                report.append(f"  Transaction Count: {pattern['transaction_count']}")
                report.append(f"  Average Amount: ₹{pattern['average_amount']:,.2f}")
                report.append(f"  Total Amount: ₹{pattern['total_amount']:,.2f}")
                report.append(f"  Date Range: {pattern['first_transaction_date']} to {pattern['last_transaction_date']}")
                report.append(f"  Days Span: {pattern['days_span']} days")
                report.append(f"  Amount Variance: {pattern['amount_variance']:.2%}")
        
        # Rapid withdrawals
        if self.detected_patterns['rapid_withdrawals']:
            report.append("\n" + "="*80)
            report.append("💨 RAPID WITHDRAWAL PATTERNS (Money In, Money Out)")
            report.append("="*80)
            for i, pattern in enumerate(self.detected_patterns['rapid_withdrawals'], 1):
                report.append(f"\nPattern #{i} [{pattern['severity'].upper()} SEVERITY]:")
                report.append(f"  Description: {pattern['description']}")
                report.append(f"  Received: ₹{pattern['received_amount']:,.2f}")
                report.append(f"  Withdrawn: ₹{pattern['withdrawn_amount']:,.2f} ({pattern['withdrawal_percentage']:.0f}%)")
                report.append(f"  Time Difference: {pattern['hours_difference']:.0f} hours")
                report.append(f"  Withdrawal Count: {pattern['withdrawal_count']}")
        
        # Round amounts
        if self.detected_patterns['round_amount_patterns']:
            report.append("\n" + "="*80)
            report.append("⭕ ROUND AMOUNT PATTERNS (Suspiciously Round Amounts)")
            report.append("="*80)
            for i, pattern in enumerate(self.detected_patterns['round_amount_patterns'], 1):
                report.append(f"  {i}. {pattern['description']}")
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("Pattern Detector - Ready for use")
    print("\nSupported patterns:")
    print("✓ Chunking patterns (multiple similar transactions)")
    print("✓ Rapid withdrawals (money received and withdrawn quickly)")
    print("✓ Round amount patterns (suspiciously round transactions)")
    print("✓ Configurable detection parameters")
    print("✓ Severity classification (high, medium, low)")
