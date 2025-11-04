"""
Transaction Matcher - Enterprise-grade interbank transaction matching
Matches transactions using UTR/RRN reference IDs with intelligent fallback strategies
Supports 4500+ banks without hardcoding
"""

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import Dict, List, Tuple, Optional, Set
import re
import warnings
warnings.filterwarnings('ignore')

class TransactionMatcher:
    """
    Matches transactions between two entities using multi-level matching strategy:
    1. Reference ID matching (UTR/RRN/Transaction ID) - Primary
    2. Amount + Date + Fuzzy Description - Secondary
    3. Amount + Date only - Tertiary fallback
    """

    def __init__(
        self, 
        date_tolerance_days: int = 3, 
        amount_tolerance_percent: float = 0.01,
        enable_reference_matching: bool = True,
        enable_fuzzy_matching: bool = True,
        min_confidence_score: float = 0.5
    ):
        """
        Initialize matcher with tolerance parameters

        Args:
            date_tolerance_days: Number of days tolerance for date matching (default: 3)
            amount_tolerance_percent: Percentage tolerance for amount matching (default: 0.01 = 1%)
            enable_reference_matching: Enable UTR/RRN reference ID matching (default: True)
            enable_fuzzy_matching: Enable fuzzy description matching (default: True)
            min_confidence_score: Minimum confidence score for a match (0.0-1.0, default: 0.5)
        """
        self.date_tolerance_days = date_tolerance_days
        self.amount_tolerance_percent = amount_tolerance_percent
        self.enable_reference_matching = enable_reference_matching
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.min_confidence_score = min_confidence_score

        # Statistics tracking
        self.stats = {
            'reference_id_matches': 0,
            'fuzzy_matches': 0,
            'amount_date_matches': 0,
            'no_match': 0,
            'low_confidence_rejected': 0
        }

    def _extract_reference_ids(self, description: str) -> Set[str]:
        """
        Extract payment system reference IDs from transaction description
        Supports UTR (NEFT/RTGS), RRN (IMPS/UPI), and other reference formats
        Works across all banks without hardcoding

        Args:
            description: Transaction description string

        Returns:
            Set of extracted reference IDs
        """
        if pd.isna(description) or not isinstance(description, str):
            return set()

        description_upper = description.upper()
        reference_ids = set()

        # Pattern 1: NEFT/RTGS UTR - 16-22 character alphanumeric starting with letters
        # Format: BANKRCYYMMDD######## (e.g., HDFCR2240101123456)
        neft_rtgs_pattern = r'\b[A-Z]{4}[A-Z0-9]{12,18}\b'

        # Pattern 2: IMPS/UPI RRN - 12 digit numeric
        imps_upi_pattern = r'\b\d{12}\b'

        # Pattern 3: Generic reference with prefix markers
        # UTR:, REF:, RRN:, TXNID:, TRANSACTION ID:, etc.
        prefix_patterns = [
            r'UTR[:\s]+([A-Z0-9]{10,25})',
            r'RRN[:\s]+(\d{10,15})',
            r'REF(?:ERENCE)?[:\s]+([A-Z0-9]{8,25})',
            r'TXN(?:\s)?ID[:\s]+([A-Z0-9]{8,25})',
            r'TRANSACTION[:\s]+ID[:\s]+([A-Z0-9]{8,25})',
            r'PAYMENT[:\s]+ID[:\s]+([A-Z0-9]{8,25})',
            r'UPIREF[:\s]+([A-Z0-9]{10,25})'
        ]

        # Extract NEFT/RTGS patterns
        for match in re.finditer(neft_rtgs_pattern, description_upper):
            ref_id = match.group().strip()
            # Validate: should not be a common word
            if len(ref_id) >= 12 and not self._is_common_word(ref_id):
                reference_ids.add(ref_id)

        # Extract IMPS/UPI patterns
        for match in re.finditer(imps_upi_pattern, description_upper):
            ref_id = match.group().strip()
            # Validate: 12 digits exactly
            if len(ref_id) == 12:
                reference_ids.add(ref_id)

        # Extract prefixed references
        for pattern in prefix_patterns:
            for match in re.finditer(pattern, description_upper):
                ref_id = match.group(1).strip()
                if len(ref_id) >= 8:  # Minimum length for valid reference
                    reference_ids.add(ref_id)

        return reference_ids

    def _is_common_word(self, text: str) -> bool:
        """Check if text is a common word (not a reference ID)"""
        common_words = {'TRANSFER', 'PAYMENT', 'DEPOSIT', 'WITHDRAWAL', 'CREDIT', 'DEBIT', 
                       'SALARY', 'INVOICE', 'REFUND', 'CHARGES', 'INTEREST'}
        return text in common_words

    def _calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """
        Calculate similarity score between two descriptions (0.0 to 1.0)
        Uses token-based matching for bank-agnostic comparison
        """
        if pd.isna(desc1) or pd.isna(desc2):
            return 0.0

        # Normalize descriptions
        desc1_clean = re.sub(r'[^a-zA-Z0-9\s]', '', str(desc1).upper())
        desc2_clean = re.sub(r'[^a-zA-Z0-9\s]', '', str(desc2).upper())

        # Extract tokens (words with length > 3)
        tokens1 = set([w for w in desc1_clean.split() if len(w) > 3])
        tokens2 = set([w for w in desc2_clean.split() if len(w) > 3])

        if not tokens1 or not tokens2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def match_transactions(
        self,
        entity1_df: pd.DataFrame,
        entity2_df: pd.DataFrame,
        entity1_name: str = "Entity 1",
        entity2_name: str = "Entity 2"
    ) -> pd.DataFrame:
        """
        Find matching transactions between two entities using multi-level strategy

        Args:
            entity1_df: DataFrame with Entity 1's transactions (must have: date, credit, debit, description)
            entity2_df: DataFrame with Entity 2's transactions (must have: date, credit, debit, description)
            entity1_name: Name of Entity 1
            entity2_name: Name of Entity 2

        Returns:
            DataFrame with matched transactions including confidence scores and match types
        """
        # Reset statistics
        self.stats = {k: 0 for k in self.stats.keys()}

        # Validate input DataFrames
        required_cols = ['date', 'credit', 'debit', 'description']
        for col in required_cols:
            if col not in entity1_df.columns or col not in entity2_df.columns:
                raise ValueError(f"Both DataFrames must contain column: {col}")

        # Convert dates to datetime - CRITICAL FIX
        entity1_df = entity1_df.copy()
        entity2_df = entity2_df.copy()
        entity1_df['date'] = pd.to_datetime(entity1_df['date'])
        entity2_df['date'] = pd.to_datetime(entity2_df['date'])

        print(f"\n🔍 Matching transactions between {entity1_name} and {entity2_name}...")
        print(f"   {entity1_name}: {len(entity1_df)} transactions")
        print(f"   {entity2_name}: {len(entity2_df)} transactions")
        print(f"   Reference ID Matching: {'✓ Enabled' if self.enable_reference_matching else '✗ Disabled'}")
        print(f"   Fuzzy Description Matching: {'✓ Enabled' if self.enable_fuzzy_matching else '✗ Disabled'}")

        matches = []
        matched_indices_entity2 = set()  # Track matched transactions to avoid duplicates

        # For each transaction in Entity 1, look for matches in Entity 2
        for idx1, row1 in entity1_df.iterrows():
            # Entity 1's credit (incoming) = Entity 2's debit (outgoing)
            if pd.notna(row1['credit']) and row1['credit'] > 0:
                match = self._find_matching_transaction(
                    row1, entity2_df, 'debit', entity1_name, entity2_name, 
                    'received', 'sent', matched_indices_entity2
                )
                if match:
                    matches.append(match)
                    matched_indices_entity2.add(match['_entity2_index'])

            # Entity 1's debit (outgoing) = Entity 2's credit (incoming)
            if pd.notna(row1['debit']) and row1['debit'] > 0:
                match = self._find_matching_transaction(
                    row1, entity2_df, 'credit', entity1_name, entity2_name, 
                    'sent', 'received', matched_indices_entity2
                )
                if match:
                    matches.append(match)
                    matched_indices_entity2.add(match['_entity2_index'])

        if not matches:
            print("   ⚠️  No matching transactions found")
            self._print_statistics()
            return pd.DataFrame()

        matches_df = pd.DataFrame(matches)
        # Remove internal tracking column
        if '_entity2_index' in matches_df.columns:
            matches_df = matches_df.drop(columns=['_entity2_index'])

        print(f"   ✅ Found {len(matches_df)} matching transactions")
        self._print_statistics()

        return matches_df

    def _find_matching_transaction(
        self,
        source_row: pd.Series,
        target_df: pd.DataFrame,
        target_column: str,
        entity1_name: str,
        entity2_name: str,
        entity1_action: str,
        entity2_action: str,
        matched_indices: Set[int]
    ) -> Optional[Dict]:
        """
        Find a matching transaction using multi-level matching strategy

        Priority:
        1. Reference ID match (if available and enabled)
        2. Amount + Date + Fuzzy description match
        3. Amount + Date match only (fallback)
        """
        source_amount = source_row['credit'] if target_column == 'debit' else source_row['debit']
        source_date = pd.to_datetime(source_row['date'])
        source_description = source_row['description']

        # Calculate date and amount ranges
        date_min = source_date - timedelta(days=self.date_tolerance_days)
        date_max = source_date + timedelta(days=self.date_tolerance_days)
        amount_tolerance = source_amount * self.amount_tolerance_percent
        amount_min = source_amount - amount_tolerance
        amount_max = source_amount + amount_tolerance

        # Filter candidates by date and amount (common for all strategies)
        candidates = target_df[
            (target_df['date'] >= date_min) &
            (target_df['date'] <= date_max) &
            (target_df[target_column] >= amount_min) &
            (target_df[target_column] <= amount_max) &
            (~target_df.index.isin(matched_indices))  # Exclude already matched
        ].copy()

        if len(candidates) == 0:
            self.stats['no_match'] += 1
            return None

        # STRATEGY 1: Reference ID Matching (Primary)
        if self.enable_reference_matching:
            source_refs = self._extract_reference_ids(source_description)
            if source_refs:
                for idx, candidate in candidates.iterrows():
                    target_refs = self._extract_reference_ids(candidate['description'])
                    common_refs = source_refs & target_refs

                    if common_refs:
                        # Found exact reference match - highest confidence
                        self.stats['reference_id_matches'] += 1
                        return self._create_match_record(
                            source_row, candidate, source_amount, source_date, 
                            target_column, entity1_name, entity2_name, 
                            entity1_action, entity2_action, idx,
                            match_type='reference_id',
                            confidence_score=1.0,
                            matched_references=list(common_refs)
                        )

        # STRATEGY 2: Fuzzy Description + Amount + Date Matching (Secondary)
        if self.enable_fuzzy_matching and len(candidates) > 0:
            best_match = None
            best_score = -1
            best_candidate = None
            best_idx = None

            for idx, candidate in candidates.iterrows():
                # Calculate similarity score
                desc_similarity = self._calculate_description_similarity(
                    source_description, candidate['description']
                )

                # Calculate date proximity score (0 to 1, where 1 is same date)
                date_diff_days = abs((pd.to_datetime(candidate['date']) - source_date).days)
                date_score = 1.0 - (date_diff_days / (self.date_tolerance_days + 1))

                # Calculate amount proximity score (0 to 1, where 1 is exact match)
                amount_diff = abs(candidate[target_column] - source_amount)
                amount_score = 1.0 - (amount_diff / (source_amount * self.amount_tolerance_percent + 0.01))
                amount_score = max(0.0, min(1.0, amount_score))

                # Weighted composite score
                # Description: 50%, Date: 25%, Amount: 25%
                composite_score = (0.5 * desc_similarity + 0.25 * date_score + 0.25 * amount_score)

                if composite_score > best_score:
                    best_score = composite_score
                    best_match = candidate
                    best_candidate = (desc_similarity, date_score, amount_score)
                    best_idx = idx

            # Check if best match meets minimum confidence threshold
            if best_match is not None and best_score >= self.min_confidence_score:
                self.stats['fuzzy_matches'] += 1
                return self._create_match_record(
                    source_row, best_match, source_amount, source_date, 
                    target_column, entity1_name, entity2_name, 
                    entity1_action, entity2_action, best_idx,
                    match_type='fuzzy_match',
                    confidence_score=best_score,
                    description_similarity=best_candidate[0],
                    date_proximity=best_candidate[1],
                    amount_proximity=best_candidate[2]
                )
            elif best_match is not None:
                self.stats['low_confidence_rejected'] += 1

        # STRATEGY 3: Amount + Date Only (Tertiary Fallback)
        # Use the closest match by date and amount when no other strategy works
        if len(candidates) > 0:
            best_match = None
            best_distance = float('inf')
            best_idx = None

            for idx, candidate in candidates.iterrows():
                date_diff = abs((pd.to_datetime(candidate['date']) - source_date).days)
                amount_diff_pct = abs(candidate[target_column] - source_amount) / source_amount * 100

                # Combined distance metric
                distance = date_diff + amount_diff_pct

                if distance < best_distance:
                    best_distance = distance
                    best_match = candidate
                    best_idx = idx

            if best_match is not None:
                # Calculate confidence based on proximity (inverse of distance)
                max_distance = self.date_tolerance_days + (self.amount_tolerance_percent * 100)
                confidence = max(0.0, 1.0 - (best_distance / max_distance))

                if confidence >= self.min_confidence_score:
                    self.stats['amount_date_matches'] += 1
                    return self._create_match_record(
                        source_row, best_match, source_amount, source_date, 
                        target_column, entity1_name, entity2_name, 
                        entity1_action, entity2_action, best_idx,
                        match_type='amount_date_only',
                        confidence_score=confidence
                    )
                else:
                    self.stats['low_confidence_rejected'] += 1

        self.stats['no_match'] += 1
        return None

    def _create_match_record(
        self,
        source_row: pd.Series,
        target_row: pd.Series,
        source_amount: float,
        source_date: datetime,
        target_column: str,
        entity1_name: str,
        entity2_name: str,
        entity1_action: str,
        entity2_action: str,
        target_idx: int,
        match_type: str,
        confidence_score: float,
        matched_references: List[str] = None,
        description_similarity: float = None,
        date_proximity: float = None,
        amount_proximity: float = None
    ) -> Dict:
        """Create a match record with comprehensive metadata"""

        target_date = pd.to_datetime(target_row['date'])
        target_amount = target_row[target_column]

        record = {
            'transaction_date': source_date.strftime('%Y-%m-%d'),
            'amount': float(source_amount),
            f'{entity1_name}_date': source_date.strftime('%Y-%m-%d'),
            f'{entity2_name}_date': target_date.strftime('%Y-%m-%d'),
            f'{entity1_name}_description': source_row['description'],
            f'{entity2_name}_description': target_row['description'],
            f'{entity1_name}_action': entity1_action,
            f'{entity2_name}_action': entity2_action,
            'date_difference_days': abs((target_date - source_date).days),
            'amount_difference': abs(target_amount - source_amount),
            'amount_difference_pct': abs(target_amount - source_amount) / source_amount * 100,
            'match_type': match_type,
            'confidence_score': round(confidence_score, 3),
            'is_high_value': source_amount > 10000,
            '_entity2_index': target_idx  # Internal tracking
        }

        # Add match-specific metadata
        if matched_references:
            record['matched_reference_ids'] = ', '.join(matched_references)

        if description_similarity is not None:
            record['description_similarity'] = round(description_similarity, 3)

        if date_proximity is not None:
            record['date_proximity_score'] = round(date_proximity, 3)

        if amount_proximity is not None:
            record['amount_proximity_score'] = round(amount_proximity, 3)

        return record

    def _print_statistics(self):
        """Print matching statistics"""
        print(f"\n📊 Matching Statistics:")
        print(f"   Reference ID Matches: {self.stats['reference_id_matches']}")
        print(f"   Fuzzy Matches: {self.stats['fuzzy_matches']}")
        print(f"   Amount+Date Matches: {self.stats['amount_date_matches']}")
        print(f"   No Match Found: {self.stats['no_match']}")
        print(f"   Low Confidence Rejected: {self.stats['low_confidence_rejected']}")

    def analyze_matches(self, matches_df: pd.DataFrame) -> Dict:
        """
        Analyze matched transactions and generate comprehensive statistics

        Returns:
            Dictionary with analysis results including confidence metrics
        """
        if len(matches_df) == 0:
            return {
                'total_transactions': 0,
                'total_value': 0.0,
                'high_value_transactions': 0,
                'high_value_total': 0.0,
                'first_transaction_date': None,
                'last_transaction_date': None,
                'average_transaction_value': 0.0,
                'date_range_days': 0,
                'average_confidence_score': 0.0
            }

        high_value_df = matches_df[matches_df['is_high_value'] == True]
        first_date = pd.to_datetime(matches_df['transaction_date']).min()
        last_date = pd.to_datetime(matches_df['transaction_date']).max()
        date_range = (last_date - first_date).days

        # Match type distribution
        match_type_counts = matches_df['match_type'].value_counts().to_dict()

        analysis = {
            'total_transactions': len(matches_df),
            'total_value': float(matches_df['amount'].sum()),
            'high_value_transactions': len(high_value_df),
            'high_value_total': float(high_value_df['amount'].sum()) if len(high_value_df) > 0 else 0.0,
            'first_transaction_date': first_date.strftime('%Y-%m-%d'),
            'last_transaction_date': last_date.strftime('%Y-%m-%d'),
            'average_transaction_value': float(matches_df['amount'].mean()),
            'date_range_days': date_range,
            'median_transaction_value': float(matches_df['amount'].median()),
            'max_transaction_value': float(matches_df['amount'].max()),
            'min_transaction_value': float(matches_df['amount'].min()),
            'average_confidence_score': float(matches_df['confidence_score'].mean()),
            'min_confidence_score': float(matches_df['confidence_score'].min()),
            'match_type_distribution': match_type_counts,
            'average_date_difference': float(matches_df['date_difference_days'].mean()),
            'max_date_difference': int(matches_df['date_difference_days'].max())
        }

        print(f"\n📊 Transaction Analysis:")
        print(f"   Total Transactions: {analysis['total_transactions']}")
        print(f"   Total Value: ₹{analysis['total_value']:,.2f}")
        print(f"   Date Range: {analysis['first_transaction_date']} to {analysis['last_transaction_date']}")
        print(f"   Average Confidence: {analysis['average_confidence_score']:.2%}")
        print(f"   Match Types: {match_type_counts}")
        print(f"   High-Value (>10K): {analysis['high_value_transactions']} transactions (₹{analysis['high_value_total']:,.2f})")

        return analysis

    def get_high_value_transactions(
        self, 
        matches_df: pd.DataFrame, 
        threshold: float = 10000
    ) -> pd.DataFrame:
        """
        Filter and return high-value transactions

        Args:
            matches_df: DataFrame with matched transactions
            threshold: Amount threshold for high-value transactions

        Returns:
            DataFrame with high-value transactions sorted by amount
        """
        if len(matches_df) == 0:
            return pd.DataFrame()

        high_value = matches_df[matches_df['amount'] > threshold].copy()
        high_value = high_value.sort_values('amount', ascending=False)

        return high_value

    def get_low_confidence_matches(
        self, 
        matches_df: pd.DataFrame, 
        threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        Get matches with confidence scores below threshold for manual review

        Args:
            matches_df: DataFrame with matched transactions
            threshold: Confidence threshold (default: 0.7)

        Returns:
            DataFrame with low confidence matches
        """
        if len(matches_df) == 0 or 'confidence_score' not in matches_df.columns:
            return pd.DataFrame()

        low_conf = matches_df[matches_df['confidence_score'] < threshold].copy()
        low_conf = low_conf.sort_values('confidence_score', ascending=True)

        return low_conf

    def generate_transaction_summary(
        self,
        matches_df: pd.DataFrame,
        entity1_name: str = "Entity 1",
        entity2_name: str = "Entity 2"
    ) -> str:
        """
        Generate a human-readable summary of matched transactions
        """
        if len(matches_df) == 0:
            return f"No transactions found between {entity1_name} and {entity2_name}."

        analysis = self.analyze_matches(matches_df)

        summary = f"""
Transaction Relationship Summary
=================================
Entities: {entity1_name} ↔ {entity2_name}

Overall Statistics:
- Total Transactions: {analysis['total_transactions']}
- Total Transaction Value: ₹{analysis['total_value']:,.2f}
- Average Transaction: ₹{analysis['average_transaction_value']:,.2f}
- Transaction Period: {analysis['first_transaction_date']} to {analysis['last_transaction_date']} ({analysis['date_range_days']} days)

Matching Quality:
- Average Confidence Score: {analysis['average_confidence_score']:.1%}
- Minimum Confidence Score: {analysis['min_confidence_score']:.1%}
- Match Type Distribution: {analysis['match_type_distribution']}

High-Value Transactions (>₹10,000):
- Count: {analysis['high_value_transactions']}
- Total Value: ₹{analysis['high_value_total']:,.2f}
- Percentage of Total: {(analysis['high_value_total'] / analysis['total_value'] * 100) if analysis['total_value'] > 0 else 0:.1f}%

Transaction Range:
- Minimum: ₹{analysis['min_transaction_value']:,.2f}
- Median: ₹{analysis['median_transaction_value']:,.2f}
- Maximum: ₹{analysis['max_transaction_value']:,.2f}

Date Alignment:
- Average Date Difference: {analysis['average_date_difference']:.1f} days
- Maximum Date Difference: {analysis['max_date_difference']} days
"""

        return summary

    def export_matches(
        self, 
        matches_df: pd.DataFrame, 
        filepath: str, 
        format: str = 'csv'
    ):
        """
        Export matched transactions to file

        Args:
            matches_df: DataFrame with matched transactions
            filepath: Output file path
            format: Export format ('csv' or 'excel')
        """
        if len(matches_df) == 0:
            print("⚠️  No matches to export")
            return

        if format.lower() == 'csv':
            matches_df.to_csv(filepath, index=False)
            print(f"✅ Exported {len(matches_df)} matches to {filepath}")
        elif format.lower() in ['excel', 'xlsx']:
            matches_df.to_excel(filepath, index=False, engine='openpyxl')
            print(f"✅ Exported {len(matches_df)} matches to {filepath}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'excel'")


# Example usage
if __name__ == "__main__":
    print("Transaction Matcher - Ready for use")
    print("\nSupported features:")
    print("✓ Multi-level matching strategy (Reference ID → Fuzzy → Amount+Date)")
    print("✓ Supports 4500+ banks without hardcoding")
    print("✓ UTR/RRN/Reference ID extraction from any bank format")
    print("✓ Confidence scoring for all matches")
    print("✓ Configurable tolerance parameters")
    print("✓ Low confidence match identification for manual review")
    print("✓ Comprehensive statistics and analysis")
