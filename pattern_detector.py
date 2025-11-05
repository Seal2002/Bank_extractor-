"""
Pattern Detector for Multi-Entity Transaction Analysis
STRONG Chunking Detection: Same/similar descriptions, same amounts, repeating patterns
Each transaction is classified as ONLY ONE pattern type (no duplicates)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from difflib import SequenceMatcher


class PatternDetector:
    """Detects suspicious transaction patterns with STRONG chunking detection"""

    def __init__(self, 
                 chunking_threshold: float = 100000,  # ₹1 lakh
                 chunking_window_days: int = 30,
                 rapid_withdrawal_hours: int = 72,
                 min_chunks: int = 3,
                 min_chunk_count: int = None,
                 similarity_threshold: float = 0.7):  # 70% description similarity
        """
        Initialize pattern detector with strong chunking detection
        """
        self.chunking_threshold = chunking_threshold
        self.chunking_window_days = chunking_window_days
        self.rapid_withdrawal_hours = rapid_withdrawal_hours
        self.similarity_threshold = similarity_threshold

        # Accept both parameter names
        if min_chunk_count is not None:
            self.min_chunks = min_chunk_count
        else:
            self.min_chunks = min_chunks

        # Track already classified transactions to avoid duplicates
        self.classified_txns = set()

    def detect_all_patterns(self, 
                           entity1_df: pd.DataFrame, 
                           entity2_df: pd.DataFrame,
                           entity1_name: str = "Entity 1",
                           entity2_name: str = "Entity 2") -> Dict[str, Any]:
        """
        Detect all suspicious patterns in transaction data
        """
        try:
            # Reset classified transactions for each detection run
            self.classified_txns = set()

            # Detect patterns in priority order
            chunking_patterns = (
                self._detect_chunking_patterns(entity1_df, entity1_name) + 
                self._detect_chunking_patterns(entity2_df, entity2_name)
            )

            # Only detect rapid withdrawals for non-chunked transactions
            rapid_withdrawals = (
                self._detect_rapid_withdrawals(entity1_df, entity1_name) + 
                self._detect_rapid_withdrawals(entity2_df, entity2_name)
            )

            patterns = {
                'chunking_patterns': chunking_patterns,
                'rapid_withdrawals': rapid_withdrawals
            }
            return patterns
        except Exception as e:
            print(f"Error in detect_all_patterns: {e}")
            return {'chunking_patterns': [], 'rapid_withdrawals': []}

    def analyze_patterns(self, 
                        entity1_df: pd.DataFrame, 
                        entity2_df: pd.DataFrame,
                        entity1_name: str = "Entity 1",
                        entity2_name: str = "Entity 2") -> Dict[str, Any]:
        """
        Alias for detect_all_patterns
        """
        return self.detect_all_patterns(entity1_df, entity2_df, entity1_name, entity2_name)

    def _mark_transactions_classified(self, txn_indices: List[int]):
        """Mark transactions as classified to prevent duplicates"""
        for idx in txn_indices:
            self.classified_txns.add(idx)

    def _is_transaction_classified(self, txn_idx: int) -> bool:
        """Check if transaction is already classified"""
        return txn_idx in self.classified_txns

    def _calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity between two descriptions (0-1)"""
        if pd.isna(desc1) or pd.isna(desc2):
            return 0.0

        desc1 = str(desc1).lower().strip()
        desc2 = str(desc2).lower().strip()

        # Exact match
        if desc1 == desc2:
            return 1.0

        # Use SequenceMatcher for partial similarity
        return SequenceMatcher(None, desc1, desc2).ratio()

    def _detect_chunking_patterns(self, df: pd.DataFrame, entity_name: str) -> List[Dict]:
        """
        STRONG Chunking Detection:
        1. Same description + same amount (multiple times)
        2. Similar description + same amount (e.g., "Payment 1", "Payment 2")
        3. Same description + similar amounts (within 5% variance)
        4. Repeating pattern: Many transactions crossing threshold
        """
        patterns = []

        if df.empty or 'debit' not in df.columns:
            return patterns

        try:
            # Filter for debit transactions (outgoing)
            debit_df = df[df['debit'].notna() & (df['debit'] > 0)].copy()

            if len(debit_df) < self.min_chunks:
                return patterns

            # Convert date to datetime
            debit_df['date'] = pd.to_datetime(debit_df['date'], errors='coerce')
            debit_df = debit_df.dropna(subset=['date']).reset_index(drop=True)

            # Sort by date and amount
            debit_df = debit_df.sort_values('date').reset_index(drop=True)

            # Get original indices
            debit_indices = df[df['debit'].notna() & (df['debit'] > 0)].index.tolist()

            detected_indices = set()

            # STRATEGY 1: Same description + Same amount (STRONGEST signal)
            desc_amount_groups = debit_df.groupby(['description', 'debit']).size().reset_index(name='count')
            desc_amount_groups = desc_amount_groups[desc_amount_groups['count'] >= self.min_chunks]

            for _, group in desc_amount_groups.iterrows():
                matching_txns = debit_df[
                    (debit_df['description'] == group['description']) & 
                    (debit_df['debit'] == group['debit'])
                ].copy()

                matching_indices = matching_txns.index.tolist()
                orig_matching_indices = [debit_indices[i] for i in matching_indices]

                # Skip if already classified
                if any(idx in self.classified_txns or idx in detected_indices for idx in orig_matching_indices):
                    continue

                total_amount = matching_txns['debit'].sum()
                if total_amount >= self.chunking_threshold:
                    date_range = matching_txns['date'].min(), matching_txns['date'].max()
                    days_span = (date_range[1] - date_range[0]).days

                    if days_span <= self.chunking_window_days:
                        pattern = {
                            'entity': entity_name,
                            'pattern_type': 'Chunking/Structuring',
                            'risk_level': 'CRITICAL',
                            'sub_type': 'Identical Transactions',
                            'description': f'🚨 IDENTICAL transactions detected: {group["count"]} payments of ₹{group["debit"]:,.0f} each (Total: ₹{total_amount:,.0f})',
                            'count': group['count'],
                            'total_amount': total_amount,
                            'avg_amount': group['debit'],
                            'max_txn': group['debit'],
                            'min_txn': group['debit'],
                            'date_range': (date_range[0].strftime('%Y-%m-%d'), date_range[1].strftime('%Y-%m-%d')),
                            'amount_range': (group['debit'], group['debit']),
                            'window_days': days_span,
                            'transactions': matching_txns[['date', 'description', 'debit', 'credit', 'balance']].reset_index(drop=True),
                            'original_indices': orig_matching_indices
                        }
                        patterns.append(pattern)
                        self._mark_transactions_classified(orig_matching_indices)
                        detected_indices.update(matching_indices)

            # STRATEGY 2: Same description + Similar amounts (within 5% variance)
            desc_groups = debit_df[~debit_df.index.isin(detected_indices)].groupby('description')

            for desc, group_df in desc_groups:
                if len(group_df) >= self.min_chunks and pd.notna(desc):
                    group_df = group_df.sort_values('debit')
                    amounts = group_df['debit'].values

                    # Check if amounts are similar (within 5% variance)
                    amount_variance = (amounts.max() - amounts.min()) / amounts.mean()

                    if amount_variance <= 0.05:  # Less than 5% variance
                        total_amount = amounts.sum()

                        if total_amount >= self.chunking_threshold:
                            matching_indices = group_df.index.tolist()
                            orig_matching_indices = [debit_indices[i] for i in matching_indices]

                            if not any(idx in self.classified_txns for idx in orig_matching_indices):
                                date_range = group_df['date'].min(), group_df['date'].max()
                                days_span = (date_range[1] - date_range[0]).days

                                if days_span <= self.chunking_window_days:
                                    pattern = {
                                        'entity': entity_name,
                                        'pattern_type': 'Chunking/Structuring',
                                        'risk_level': 'HIGH',
                                        'sub_type': 'Similar Amounts (Same Desc)',
                                        'description': f'Multiple payments with same description & similar amounts: {len(group_df)} transactions totaling ₹{total_amount:,.0f}',
                                        'count': len(group_df),
                                        'total_amount': total_amount,
                                        'avg_amount': amounts.mean(),
                                        'max_txn': amounts.max(),
                                        'min_txn': amounts.min(),
                                        'date_range': (date_range[0].strftime('%Y-%m-%d'), date_range[1].strftime('%Y-%m-%d')),
                                        'amount_range': (amounts.min(), amounts.max()),
                                        'window_days': days_span,
                                        'variance_percent': amount_variance * 100,
                                        'transactions': group_df[['date', 'description', 'debit', 'credit', 'balance']].reset_index(drop=True),
                                        'original_indices': orig_matching_indices
                                    }
                                    patterns.append(pattern)
                                    self._mark_transactions_classified(orig_matching_indices)
                                    detected_indices.update(matching_indices)

            # STRATEGY 3: Similar descriptions + Same amount
            remaining_df = debit_df[~debit_df.index.isin(detected_indices)].copy()
            amount_groups = remaining_df[remaining_df['debit'] > 50000].groupby('debit')

            for amount, amount_group in amount_groups:
                if len(amount_group) >= self.min_chunks:
                    # Check if descriptions are similar
                    descriptions = amount_group['description'].unique()

                    similar_desc_indices = []
                    for i, desc1 in enumerate(descriptions):
                        similar_count = 1
                        for desc2 in descriptions[i+1:]:
                            similarity = self._calculate_description_similarity(str(desc1), str(desc2))
                            if similarity >= self.similarity_threshold:
                                similar_count += 1

                        if similar_count >= self.min_chunks:
                            # Get all transactions with similar descriptions
                            for desc in descriptions:
                                if self._calculate_description_similarity(str(desc1), str(desc)) >= self.similarity_threshold:
                                    matching = amount_group[amount_group['description'] == desc]
                                    similar_desc_indices.extend(matching.index.tolist())

                    if len(similar_desc_indices) >= self.min_chunks:
                        similar_desc_indices = list(set(similar_desc_indices))  # Remove duplicates
                        similar_txns = remaining_df.loc[similar_desc_indices]
                        total_amount = similar_txns['debit'].sum()

                        if total_amount >= self.chunking_threshold:
                            orig_matching_indices = [debit_indices[i] for i in similar_desc_indices]

                            if not any(idx in self.classified_txns for idx in orig_matching_indices):
                                date_range = similar_txns['date'].min(), similar_txns['date'].max()
                                days_span = (date_range[1] - date_range[0]).days

                                if days_span <= self.chunking_window_days:
                                    pattern = {
                                        'entity': entity_name,
                                        'pattern_type': 'Chunking/Structuring',
                                        'risk_level': 'MEDIUM',
                                        'sub_type': 'Similar Descriptions & Amounts',
                                        'description': f'Transactions with similar descriptions & same amount: {len(similar_desc_indices)} payments of ₹{amount:,.0f}',
                                        'count': len(similar_desc_indices),
                                        'total_amount': total_amount,
                                        'avg_amount': similar_txns['debit'].mean(),
                                        'max_txn': similar_txns['debit'].max(),
                                        'min_txn': similar_txns['debit'].min(),
                                        'date_range': (date_range[0].strftime('%Y-%m-%d'), date_range[1].strftime('%Y-%m-%d')),
                                        'amount_range': (similar_txns['debit'].min(), similar_txns['debit'].max()),
                                        'window_days': days_span,
                                        'transactions': similar_txns[['date', 'description', 'debit', 'credit', 'balance']].reset_index(drop=True),
                                        'original_indices': orig_matching_indices
                                    }
                                    patterns.append(pattern)
                                    self._mark_transactions_classified(orig_matching_indices)
                                    detected_indices.update(similar_desc_indices)

        except Exception as e:
            print(f"Error in chunking detection: {e}")

        return patterns

    def _detect_rapid_withdrawals(self, df: pd.DataFrame, entity_name: str) -> List[Dict]:
        """
        Detect rapid withdrawal patterns: Money withdrawn soon after deposit
        Only for transactions NOT already classified as chunking
        """
        patterns = []

        if df.empty or 'credit' not in df.columns or 'debit' not in df.columns:
            return patterns

        try:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
            df_copy = df_copy.dropna(subset=['date']).reset_index(drop=True)
            df_copy = df_copy.sort_values('date').reset_index(drop=True)

            # Get significant credit transactions (receipts)
            significant_credits = df_copy[
                (df_copy['credit'].notna()) & 
                (df_copy['credit'] > 50000)
            ].copy()

            detected_credit_indices = set()

            for credit_idx, credit_txn in significant_credits.iterrows():
                # Skip if already classified
                if credit_idx in self.classified_txns or credit_idx in detected_credit_indices:
                    continue

                receipt_date = credit_txn['date']
                receipt_amount = credit_txn['credit']

                # Look for debits within the rapid withdrawal window
                window_start = receipt_date
                window_end = receipt_date + timedelta(hours=self.rapid_withdrawal_hours)

                debits_in_window = df_copy[
                    (df_copy['date'] > receipt_date) &
                    (df_copy['date'] <= window_end) &
                    (df_copy['debit'].notna()) &
                    (df_copy['debit'] > 5000)
                ].copy()

                # Filter out already classified debits
                unclassified_debits = debits_in_window[
                    ~debits_in_window.index.isin(self.classified_txns)
                ].copy()

                if len(unclassified_debits) > 0:
                    total_withdrawn = unclassified_debits['debit'].sum()
                    withdrawal_percentage = (total_withdrawn / receipt_amount) * 100

                    if withdrawal_percentage >= 40:
                        hours_elapsed = (unclassified_debits['date'].min() - receipt_date).total_seconds() / 3600

                        # Determine risk level
                        if withdrawal_percentage >= 90:
                            risk_level = 'CRITICAL'
                        elif withdrawal_percentage >= 70:
                            risk_level = 'HIGH'
                        else:
                            risk_level = 'MEDIUM'

                        pattern_key = (credit_idx, receipt_date, receipt_amount)
                        if pattern_key not in detected_credit_indices:
                            pattern = {
                                'entity': entity_name,
                                'pattern_type': 'Rapid Withdrawal',
                                'risk_level': risk_level,
                                'description': f'₹{withdrawal_percentage:.0f}% of deposit (₹{receipt_amount:,.0f}) withdrawn within {hours_elapsed:.1f} hours',
                                'received_amount': receipt_amount,
                                'withdrawn_amount': total_withdrawn,
                                'withdrawal_percentage': withdrawal_percentage,
                                'receipt_date': receipt_date.strftime('%Y-%m-%d %H:%M'),
                                'withdrawal_start': unclassified_debits['date'].min().strftime('%Y-%m-%d %H:%M'),
                                'withdrawal_end': unclassified_debits['date'].max().strftime('%Y-%m-%d %H:%M'),
                                'hours_elapsed': hours_elapsed,
                                'withdrawal_count': len(unclassified_debits),
                                'receipt_transaction': pd.DataFrame([credit_txn]),
                                'withdrawal_transactions': unclassified_debits[['date', 'description', 'debit', 'balance']].reset_index(drop=True),
                                'original_indices': [credit_idx] + unclassified_debits.index.tolist()
                            }
                            patterns.append(pattern)
                            self._mark_transactions_classified([credit_idx] + unclassified_debits.index.tolist())
                            detected_credit_indices.add(pattern_key)

        except Exception as e:
            print(f"Error in rapid withdrawal detection: {e}")

        return patterns
