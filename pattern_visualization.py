"""
Pattern Visualization for Multi-Entity Transaction Analysis
Displays detected patterns in Streamlit interface
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


def display_all_patterns(patterns: Dict[str, Any], 
                        matches_df: pd.DataFrame,
                        entity1_data: Dict,
                        entity2_data: Dict,
                        entity1_name: str,
                        entity2_name: str):
    """
    Display all detected patterns in Rule-Based Transaction Analysis section
    with Pattern Analysis and Transaction Details subsections
    """

    # Create tabs for Pattern Analysis and Transaction Details
    pattern_tab, details_tab = st.tabs(["📊 Pattern Analysis", "📋 Transaction Details"])

    # ========== PATTERN ANALYSIS TAB ==========
    with pattern_tab:
        st.markdown("### Detected Patterns")

        # Display pattern summary
        pattern_count = 0
        if patterns.get('chunking_patterns'):
            pattern_count += len(patterns['chunking_patterns'])
        if patterns.get('rapid_withdrawals'):
            pattern_count += len(patterns['rapid_withdrawals'])

        if pattern_count > 0:
            st.warning(f"⚠️ **{pattern_count} suspicious pattern(s) detected**")
        else:
            st.success("✅ No suspicious patterns detected")

        st.markdown("---")

        # 1. CHUNKING PATTERNS
        if patterns.get('chunking_patterns') and len(patterns['chunking_patterns']) > 0:
            _display_chunking_patterns(patterns['chunking_patterns'])

        # 2. RAPID WITHDRAWAL PATTERNS
        if patterns.get('rapid_withdrawals') and len(patterns['rapid_withdrawals']) > 0:
            _display_rapid_withdrawal_patterns(patterns['rapid_withdrawals'])

    # ========== TRANSACTION DETAILS TAB ==========
    with details_tab:
        st.markdown("### Matched Transaction Details")
        st.markdown("""
        This section displays all matched transactions between entities with pattern information.
        """)

        if matches_df is not None and len(matches_df) > 0:
            # Filter and sorting options
            col1, col2, col3 = st.columns(3)

            with col1:
                show_high_value = st.checkbox("High-value (≥10K)", value=False, key="high_value_filter_detail")

            with col2:
                sort_option = st.selectbox(
                    "Sort by",
                    ["Date (Newest)", "Date (Oldest)", "Amount (High→Low)", "Amount (Low→High)"],
                    key="sort_option_detail"
                )

            with col3:
                st.write("")  # Spacer for alignment

            # Apply filters
            filtered_df = matches_df.copy()
            if show_high_value:
                filtered_df = filtered_df[filtered_df.get('amount', filtered_df.get('match_amount', 0)) >= 10000]

            # Apply sorting
            date_col = 'transaction_date' if 'transaction_date' in filtered_df.columns else 'date'
            amount_col = 'amount' if 'amount' in filtered_df.columns else 'match_amount'

            if sort_option == "Date (Newest)":
                filtered_df = filtered_df.sort_values(date_col, ascending=False)
            elif sort_option == "Date (Oldest)":
                filtered_df = filtered_df.sort_values(date_col, ascending=True)
            elif sort_option == "Amount (High→Low)":
                filtered_df = filtered_df.sort_values(amount_col, ascending=False)
            elif sort_option == "Amount (Low→High)":
                filtered_df = filtered_df.sort_values(amount_col, ascending=True)

            # Prepare display dataframe
            display_df = filtered_df.copy()

            # Select columns for display
            display_cols = []

            # Add date column
            if 'transaction_date' in display_df.columns:
                display_cols.append('transaction_date')
            elif 'date' in display_df.columns:
                display_cols.append('date')

            # Add amount column
            if 'amount' in display_df.columns:
                display_cols.append('amount')
            elif 'match_amount' in display_df.columns:
                display_cols.append('match_amount')

            # Add confidence
            if 'confidence' in display_df.columns:
                display_cols.append('confidence')
            elif 'match_confidence' in display_df.columns:
                display_cols.append('match_confidence')

            # Add entity descriptions
            entity1_desc_col = None
            entity2_desc_col = None

            for col in display_df.columns:
                if 'entity1' in col.lower() and 'description' in col.lower():
                    entity1_desc_col = col
                if 'entity2' in col.lower() and 'description' in col.lower():
                    entity2_desc_col = col

            if entity1_desc_col and entity1_desc_col not in display_cols:
                display_cols.append(entity1_desc_col)
            if entity2_desc_col and entity2_desc_col not in display_cols:
                display_cols.append(entity2_desc_col)

            # Fallback: add any remaining useful columns
            if len(display_cols) < 5:
                for col in display_df.columns:
                    if col not in display_cols and col not in ['index', 'level_0']:
                        display_cols.append(col)
                        if len(display_cols) >= 6:
                            break

            # Display the table
            display_df_final = display_df[display_cols] if display_cols else display_df

            # Create column configuration
            column_config = {}
            for col in display_cols:
                if col == 'transaction_date' or col == 'date':
                    column_config[col] = st.column_config.DateColumn("Date", format="DD/MM/YYYY")
                elif col == 'amount' or col == 'match_amount':
                    column_config[col] = st.column_config.NumberColumn("Amount", format="₹%.0f")
                elif col == 'confidence' or col == 'match_confidence':
                    column_config[col] = st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100)

            # Display dataframe
            st.dataframe(
                display_df_final,
                width='stretch',
                column_config=column_config if column_config else None,
                height=400
            )

            # Display statistics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Transactions", len(filtered_df))
            with col2:
                total_value = filtered_df[amount_col].sum() if amount_col in filtered_df.columns else 0
                st.metric("Total Value", f"₹{total_value:,.0f}")
            with col3:
                if 'confidence' in filtered_df.columns or 'match_confidence' in filtered_df.columns:
                    conf_col = 'confidence' if 'confidence' in filtered_df.columns else 'match_confidence'
                    avg_conf = filtered_df[conf_col].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            with col4:
                st.metric("Filters Applied", f"{'High-value' if show_high_value else 'All'}")

        else:
            st.info("📭 No matched transactions found to display")

    st.markdown("---")


def _display_chunking_patterns(chunking_patterns: List[Dict]):
    """Display chunking/structuring patterns"""

    st.markdown("#### 🔴 Chunking Patterns Detected")
    st.markdown("""
    **Pattern Type:** Transaction Chunking/Structuring

    **Description:** Large amounts split into smaller transactions to avoid detection thresholds
    """)

    for idx, pattern in enumerate(chunking_patterns, 1):
        risk_emoji = "🔴" if pattern.get('risk_level') == 'HIGH' else "🟡"

        with st.expander(
            f"{risk_emoji} Pattern #{idx} - {pattern['entity']} ({pattern['count']} txns, ₹{pattern['total_amount']:,.0f})", 
            expanded=(idx == 1)
        ):
            # Risk level and description
            if pattern.get('risk_level'):
                st.markdown(f"**Risk Level:** {pattern['risk_level']}")
            if pattern.get('description'):
                st.info(pattern['description'])

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Count", pattern['count'])
            with col2:
                st.metric("Total", f"₹{pattern['total_amount']:,.0f}")
            with col3:
                st.metric("Average", f"₹{pattern['avg_amount']:,.0f}")
            with col4:
                st.metric("Max Single", f"₹{pattern['max_txn']:,.0f}")

            # Details
            st.markdown("**Pattern Details:**")
            details_text = f"""
- **Date Range:** {pattern['date_range'][0]} to {pattern['date_range'][1]}
- **Amount Range:** ₹{pattern['amount_range'][0]:,.0f} - ₹{pattern['amount_range'][1]:,.0f}
- **Time Window:** {pattern['window_days']} days
- **Min Transaction:** ₹{pattern['min_txn']:,.0f}
            """
            st.markdown(details_text)

            # Display transactions
            st.markdown("**Transactions in Pattern:**")
            if isinstance(pattern.get('transactions'), pd.DataFrame) and len(pattern['transactions']) > 0:
                txn_display = pattern['transactions'].copy()
                st.dataframe(
                    txn_display,
                    width='stretch',
                    column_config={
                        "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "debit": st.column_config.NumberColumn("Debit", format="₹%.0f"),
                        "credit": st.column_config.NumberColumn("Credit", format="₹%.0f"),
                        "balance": st.column_config.NumberColumn("Balance", format="₹%.0f")
                    }
                )

            st.markdown("---")


def _display_rapid_withdrawal_patterns(rapid_withdrawals: List[Dict]):
    """Display rapid withdrawal patterns"""

    st.markdown("#### 🔴 Rapid Withdrawal Patterns Detected")
    st.markdown("""
    **Pattern Type:** Rapid Withdrawal After Receipt

    **Description:** Large amounts withdrawn shortly after being received (possible layering/structuring)
    """)

    for idx, pattern in enumerate(rapid_withdrawals, 1):
        risk_emoji = "🔴" if pattern.get('risk_level') in ['CRITICAL', 'HIGH'] else "🟡"

        with st.expander(
            f"{risk_emoji} Pattern #{idx} - {pattern['entity']} ({pattern['withdrawal_percentage']:.0f}% withdrawn in {pattern['hours_elapsed']:.1f}h)", 
            expanded=(idx == 1)
        ):
            # Risk level and description
            if pattern.get('risk_level'):
                st.markdown(f"**Risk Level:** {pattern['risk_level']}")
            if pattern.get('description'):
                st.info(pattern['description'])

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Received", f"₹{pattern['received_amount']:,.0f}")
            with col2:
                st.metric("Withdrawn", f"₹{pattern['withdrawn_amount']:,.0f}")
            with col3:
                st.metric("% Withdrawn", f"{pattern['withdrawal_percentage']:.1f}%")
            with col4:
                st.metric("Time", f"{pattern['hours_elapsed']:.1f}h")

            # Timeline
            st.markdown("**Transaction Timeline:**")
            timeline = f"""
- **Receipt Date:** {pattern['receipt_date']}
- **Withdrawal Start:** {pattern['withdrawal_start']}
- **Withdrawal End:** {pattern['withdrawal_end']}
- **Withdrawal Count:** {pattern['withdrawal_count']}
            """
            st.markdown(timeline)

            # Transactions
            st.markdown("**Receipt Transaction:**")
            if isinstance(pattern.get('receipt_transaction'), pd.DataFrame) and len(pattern['receipt_transaction']) > 0:
                receipt_df = pattern['receipt_transaction'][['date', 'description', 'credit', 'balance']].copy()
                st.dataframe(
                    receipt_df,
                    width='stretch',
                    column_config={
                        "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "credit": st.column_config.NumberColumn("Amount Received", format="₹%.0f"),
                        "balance": st.column_config.NumberColumn("Balance", format="₹%.0f")
                    }
                )

            st.markdown("**Withdrawal Transactions:**")
            if isinstance(pattern.get('withdrawal_transactions'), pd.DataFrame) and len(pattern['withdrawal_transactions']) > 0:
                withdrawal_df = pattern['withdrawal_transactions'][['date', 'description', 'debit', 'balance']].copy()
                st.dataframe(
                    withdrawal_df,
                    width='stretch',
                    column_config={
                        "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "debit": st.column_config.NumberColumn("Amount Withdrawn", format="₹%.0f"),
                        "balance": st.column_config.NumberColumn("Balance", format="₹%.0f")
                    }
                )

            st.markdown("---")
