"""
Simplified Dashboard - Phase 2: Multi-Entity Transaction Analysis
Easy-to-understand visualizations with CSV download options
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import sys
# ============ API KEY CONFIGURATION ============
# Get Google API key from Streamlit secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
if not GOOGLE_API_KEY:
    st.error("⚠️ Google API Key not found! Please add GOOGLE_API_KEY to .streamlit/secrets.toml")
    st.stop()

# Set environment variable for downstream code
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# ===============================================

sys.path.insert(0, "/mnt/project")
from core.bank_statement_extractor import BankStatementExtractor
from core.transaction_matcher import TransactionMatcher

# Page configuration
st.set_page_config(
    page_title="Multi-Entity Transaction Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
    }
    
    /* Simplify download buttons */
    .stDownloadButton button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'entity1_data' not in st.session_state:
    st.session_state.entity1_data = None
if 'entity2_data' not in st.session_state:
    st.session_state.entity2_data = None
if 'entity1_name' not in st.session_state:
    st.session_state.entity1_name = "Entity 1"
if 'entity2_name' not in st.session_state:
    st.session_state.entity2_name = "Entity 2"
if 'matches_df' not in st.session_state:
    st.session_state.matches_df = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = None

def process_statement(uploaded_file, entity_name):
    """Process uploaded bank statement"""
    try:
        # Get current working directory
        current_dir = os.getcwd()
        
        # Create absolute paths for directories
        temp_dir = os.path.join(current_dir, "temp")
        extracted_dir = os.path.join(current_dir, "extracted_data")
        
        # Create directories with absolute paths
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(extracted_dir, exist_ok=True)
        
        # Handle CSV files
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['date', 'description', 'debit', 'credit', 'balance']
            if not all(col in df.columns for col in required_cols):
                return {
                    'success': False,
                    'error': f"CSV must contain columns: {', '.join(required_cols)}"
                }
            
            # Ensure transaction_id column exists
            if 'transaction_id' not in df.columns:
                df['transaction_id'] = None
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['date'])
            
            return {
                'success': True,
                'transactions': df,
                'metadata': {
                    'account_holder_name': entity_name,
                    'source': uploaded_file.name,
                    'bank_name': entity_name,
                    'account_number': 'N/A'
                },
                'transaction_count': len(df),
                'source_file': uploaded_file.name
            }
        
        # Handle PDF files
        else:
            # Use absolute path for temp file
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize extractor
            extractor = BankStatementExtractor()
            
            # Extract data
            result = extractor.extract(temp_path, output_dir=extracted_dir)
            
            # Load extracted CSV data
            transactions_df = pd.read_csv(result['transactions_file'])
            
            # Ensure dates are datetime objects
            transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors='coerce')
            
            # Remove any rows with invalid dates
            transactions_df = transactions_df.dropna(subset=['date'])
            
            # Load metadata
            metadata = {}
            if result['metadata_file'] and os.path.exists(result['metadata_file']):
                with open(result['metadata_file'], 'r') as f:
                    metadata = json.load(f)
            
            # Update metadata with entity name if not present
            if not metadata.get('account_holder_name'):
                metadata['account_holder_name'] = entity_name
            
            # Clean up temp PDF file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return {
                'success': True,
                'transactions': transactions_df,
                'metadata': metadata,
                'transaction_count': result['transaction_count'],
                'source_file': uploaded_file.name,
                'csv_file': result['transactions_file']
            }
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error processing statement: {error_detail}")
        return {
            'success': False,
            'error': f"{str(e)}\n\nDetails: Check console for full error trace."
        }

def display_entity_summary(entity_data, entity_name, col):
    """Display simplified summary for one entity with metadata"""
    with col:
        st.markdown(f"### 📊 {entity_name}")
        
        if entity_data is None:
            st.info("No data uploaded yet")
            return
        
        df = entity_data['transactions']
        metadata = entity_data.get('metadata', {})
        
        # Account info in clean format
        st.markdown("**📋 Account Details**")
        account_holder = metadata.get('account_holder_name', 'N/A')
        account_number = metadata.get('account_number', 'N/A')
        bank_name = metadata.get('bank_name', 'N/A')
        branch = metadata.get('branch', 'N/A')
        
        # Display in clean format
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.text(f"👤 {account_holder}")
            st.text(f"🏦 {bank_name}")
        with info_col2:
            st.text(f"🔢 {account_number}")
            if branch != 'N/A':
                st.text(f"📍 {branch}")
        
        # Statement period and balances
        st.markdown("---")
        st.markdown("**💰 Statement Period**")
        
        period_from = metadata.get('statement_period_from', 'N/A')
        period_to = metadata.get('statement_period_to', 'N/A')
        opening_balance = metadata.get('opening_balance', 'N/A')
        closing_balance = metadata.get('closing_balance', 'N/A')
        currency = metadata.get('currency', 'INR')
        
        if period_from != 'N/A' and period_to != 'N/A':
            st.text(f"📅 {period_from} to {period_to}")
        
        balance_col1, balance_col2 = st.columns(2)
        with balance_col1:
            if opening_balance != 'N/A':
                try:
                    opening_val = float(opening_balance)
                    st.text(f"Opening: {currency} {opening_val:,.2f}")
                except:
                    st.text(f"Opening: {opening_balance}")
        with balance_col2:
            if closing_balance != 'N/A':
                try:
                    closing_val = float(closing_balance)
                    st.text(f"Closing: {currency} {closing_val:,.2f}")
                except:
                    st.text(f"Closing: {closing_balance}")
        
        st.markdown("---")
        
        # Simple key metrics
        st.markdown("**📊 Transaction Summary**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Transactions", len(df))
            st.metric("Total Credits", f"₹{df['credit'].sum():,.0f}")
        
        with col2:
            # Date range
            try:
                date_col = pd.to_datetime(df['date'])
                min_date = date_col.min()
                max_date = date_col.max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    days = (max_date - min_date).days
                    st.metric("Period (days)", days)
                else:
                    st.metric("Period (days)", "N/A")
            except:
                st.metric("Period (days)", "N/A")
            
            st.metric("Total Debits", f"₹{df['debit'].sum():,.0f}")
        
        # Add download buttons for this entity
        st.markdown("---")
        st.markdown("**📥 Download Options**")
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            # Download CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 Transactions CSV",
                data=csv_data,
                file_name=f"{entity_name.replace(' ', '_')}_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width='stretch',
                key=f"download_csv_{entity_name}"
            )
        
        with dl_col2:
            # Download Metadata JSON
            metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
            st.download_button(
                label="📋 Metadata JSON",
                data=metadata_json,
                file_name=f"{entity_name.replace(' ', '_')}_metadata_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                width='stretch',
                key=f"download_metadata_{entity_name}"
            )

def create_simple_flow_chart(matches_df, entity1_name, entity2_name):
    """Create simple bar chart showing money flow"""
    if len(matches_df) == 0:
        return None
    
    # Calculate flows
    sent_by_e1 = matches_df[matches_df[f'{entity1_name}_action'] == 'sent']['amount'].sum()
    received_by_e1 = matches_df[matches_df[f'{entity1_name}_action'] == 'received']['amount'].sum()
    
    # Create simple bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f'{entity1_name} → {entity2_name}', f'{entity2_name} → {entity1_name}'],
        y=[sent_by_e1, received_by_e1],
        text=[f'₹{sent_by_e1:,.0f}', f'₹{received_by_e1:,.0f}'],
        textposition='auto',
        textfont=dict(size=14, color='white'),
        marker=dict(
            color=['#3498db', '#2ecc71'],
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"💸 Money Flow: Who Paid Whom?",
            font=dict(size=18, family="Arial", color="#2c3e50", weight='bold')
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=13)
        ),
        yaxis=dict(
            title="Amount (₹)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickformat=',.0f'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        font=dict(size=13),
        showlegend=False
    )
    
    return fig

def create_simple_timeline(matches_df):
    """Create simple line chart of transactions over time"""
    if len(matches_df) == 0:
        return None
    
    timeline_df = matches_df.copy()
    timeline_df['transaction_date'] = pd.to_datetime(timeline_df['transaction_date'])
    timeline_df = timeline_df.sort_values('transaction_date')
    
    fig = go.Figure()
    
    # Simple line chart
    fig.add_trace(go.Scatter(
        x=timeline_df['transaction_date'],
        y=timeline_df['amount'],
        mode='lines+markers',
        name='Transactions',
        line=dict(color='#3498db', width=3),
        marker=dict(
            size=10,
            color='#e74c3c',
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Amount:</b> ₹%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="📅 Transactions Over Time",
            font=dict(size=18, family="Arial", color="#2c3e50", weight='bold')
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="Amount (₹)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickformat=',.0f'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        hovermode='closest'
    )
    
    return fig

def main():
    st.title("📄 Multi-Entity Transaction Analyzer")
    st.markdown('**Simplified Dashboard** - Easy to understand visualizations', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏢 Entity Configuration")
        
        # Entity names
        entity1_name = st.text_input("Entity 1 Name", value=st.session_state.entity1_name)
        entity2_name = st.text_input("Entity 2 Name", value=st.session_state.entity2_name)
        
        if entity1_name != st.session_state.entity1_name or entity2_name != st.session_state.entity2_name:
            st.session_state.entity1_name = entity1_name
            st.session_state.entity2_name = entity2_name
            st.rerun()
        
        st.markdown("---")
        st.markdown("## 📤 Upload Statements")
        
        # Entity 1 upload
        st.markdown(f"### {entity1_name}")
        entity1_file = st.file_uploader(
            f"Upload statement for {entity1_name}",
            type=['pdf', 'csv'],
            key='entity1',
            help="Upload PDF bank statement or CSV file"
        )
        
        if entity1_file and st.button(f"Process {entity1_name}", key='process1'):
            with st.spinner(f"Processing {entity1_name}'s statement..."):
                result = process_statement(entity1_file, entity1_name)
                if result['success']:
                    st.session_state.entity1_data = result
                    st.success(f"✅ {entity1_name} data loaded: {result['transaction_count']} transactions")
                else:
                    st.error(f"❌ Error: {result['error']}")
        
        st.markdown("---")
        
        # Entity 2 upload
        st.markdown(f"### {entity2_name}")
        entity2_file = st.file_uploader(
            f"Upload statement for {entity2_name}",
            type=['pdf', 'csv'],
            key='entity2',
            help="Upload PDF bank statement or CSV file"
        )
        
        if entity2_file and st.button(f"Process {entity2_name}", key='process2'):
            with st.spinner(f"Processing {entity2_name}'s statement..."):
                result = process_statement(entity2_file, entity2_name)
                if result['success']:
                    st.session_state.entity2_data = result
                    st.success(f"✅ {entity2_name} data loaded: {result['transaction_count']} transactions")
                else:
                    st.error(f"❌ Error: {result['error']}")
        
        st.markdown("---")
        
        # Match transactions button
        if st.session_state.entity1_data and st.session_state.entity2_data:
            if st.button("🔍 Find Matching Transactions", type="primary", width='stretch'):
                with st.spinner("Finding matching transactions..."):
                    entity1_df = st.session_state.entity1_data['transactions'].copy()
                    entity2_df = st.session_state.entity2_data['transactions'].copy()
                    
                    # Remove any invalid dates
                    entity1_df = entity1_df.dropna(subset=['date'])
                    entity2_df = entity2_df.dropna(subset=['date'])
                    
                    matcher = TransactionMatcher()
                    matches = matcher.match_transactions(
                        entity1_df,
                        entity2_df,
                        entity1_name,
                        entity2_name
                    )
                    st.session_state.matches_df = matches
                    st.session_state.analysis = matcher.analyze_matches(matches)
                    
                    if len(matches) == 0:
                        st.warning(f"⚠️ No matching transactions found")
                    else:
                        st.success(f"✅ Found {len(matches)} matching transactions")
                    
                    st.rerun()
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset All", width='stretch'):
            st.session_state.entity1_data = None
            st.session_state.entity2_data = None
            st.session_state.matches_df = None
            st.session_state.analysis = None
            st.rerun()
    
    # Main content
    if st.session_state.entity1_data or st.session_state.entity2_data:
        # Show entity summaries
        col1, col2 = st.columns(2)
        display_entity_summary(st.session_state.entity1_data, entity1_name, col1)
        display_entity_summary(st.session_state.entity2_data, entity2_name, col2)
        
        # Optional: Detailed metadata viewer
        if st.session_state.entity1_data or st.session_state.entity2_data:
            with st.expander("🔍 View Detailed Metadata", expanded=False):
                st.markdown("### 📋 Complete Metadata Information")
                st.markdown("This section shows all metadata extracted from the bank statements.")
                
                meta_col1, meta_col2 = st.columns(2)
                
                with meta_col1:
                    if st.session_state.entity1_data:
                        st.markdown(f"#### {entity1_name}")
                        metadata1 = st.session_state.entity1_data.get('metadata', {})
                        
                        if metadata1:
                            # Display as formatted JSON
                            st.json(metadata1)
                        else:
                            st.info("No metadata available")
                
                with meta_col2:
                    if st.session_state.entity2_data:
                        st.markdown(f"#### {entity2_name}")
                        metadata2 = st.session_state.entity2_data.get('metadata', {})
                        
                        if metadata2:
                            # Display as formatted JSON
                            st.json(metadata2)
                        else:
                            st.info("No metadata available")
        
        st.markdown("---")
        
        # Show matching results
        if st.session_state.matches_df is not None:
            matches_df = st.session_state.matches_df
            analysis = st.session_state.analysis
            
            if len(matches_df) == 0:
                # No matches found
                st.markdown("## ⚠️ No Matching Transactions Found")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **{entity1_name}**: {len(st.session_state.entity1_data['transactions'])} transactions
                    
                    No matching transactions were found between the two entities.
                    """)
                
                with col2:
                    st.info(f"""
                    **{entity2_name}**: {len(st.session_state.entity2_data['transactions'])} transactions
                    
                    **Possible reasons:**
                    - Different time periods
                    - No common transactions
                    - Different transaction amounts
                    """)
            
            else:
                # Matches found - show analysis
                st.markdown("## 📊 Transaction Analysis Results")
                
                # Key metrics in simple format
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Matches", analysis['total_transactions'])
                with col2:
                    st.metric("Total Value", f"₹{analysis['total_value']:,.0f}")
                with col3:
                    st.metric("Average Transaction", f"₹{analysis['average_transaction_value']:,.0f}")
                with col4:
                    st.metric("High-Value (>₹10K)", analysis['high_value_transactions'])
                
                st.markdown("---")
                
                # Simple visualizations in tabs
                st.markdown("## 📈 Visual Analysis")
                
                tab1, tab2 = st.tabs([
                    "💸 Money Flow",
                    "📅 Timeline", 
                ])
                
                with tab1:
                    fig = create_simple_flow_chart(matches_df, entity1_name, entity2_name)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                        st.markdown("""
                        **What this shows:** This chart shows who paid money to whom. 
                        The blue bar shows money sent from left to right, 
                        the green bar shows money sent from right to left.
                        """)
                
                with tab2:
                    fig = create_simple_timeline(matches_df)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                        st.markdown("""
                        **What this shows:** This chart shows when transactions happened 
                        and how much was paid on each date.
                        """)
                
                st.markdown("---")
                
                # Transaction table
                st.markdown("## 📋 Transaction Details")
                
                # Simple filters
                col1, col2 = st.columns(2)
                
                with col1:
                    show_high_value = st.checkbox("Show only high-value (>₹10K)", value=False)
                
                with col2:
                    sort_option = st.selectbox(
                        "Sort by",
                        ["Date (Newest First)", "Date (Oldest First)", "Amount (Highest First)", "Amount (Lowest First)"]
                    )
                
                # Apply filters
                filtered_df = matches_df.copy()
                
                if show_high_value:
                    filtered_df = filtered_df[filtered_df['is_high_value'] == True]
                
                # Apply sorting
                if sort_option == "Date (Newest First)":
                    filtered_df = filtered_df.sort_values('transaction_date', ascending=False)
                elif sort_option == "Date (Oldest First)":
                    filtered_df = filtered_df.sort_values('transaction_date', ascending=True)
                elif sort_option == "Amount (Highest First)":
                    filtered_df = filtered_df.sort_values('amount', ascending=False)
                else:
                    filtered_df = filtered_df.sort_values('amount', ascending=True)
                
                # Display simplified table
                display_cols = ['transaction_date', 'amount', f'{entity1_name}_action', f'{entity1_name}_description', f'{entity2_name}_description']
                st.dataframe(
                    filtered_df[display_cols],
                    width='stretch',
                    column_config={
                        "transaction_date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
                        f"{entity1_name}_action": st.column_config.TextColumn(f"{entity1_name} Action"),
                        f"{entity1_name}_description": st.column_config.TextColumn(f"{entity1_name} Description"),
                        f"{entity2_name}_description": st.column_config.TextColumn(f"{entity2_name} Description")
                    }
                )
                
                # Download section
                st.markdown("---")
                st.markdown("## 📥 Download Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download all matches
                    csv_all = matches_df.to_csv(index=False)
                    st.download_button(
                        label="📄 All Matches CSV",
                        data=csv_all,
                        file_name=f"matched_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
                
                with col2:
                    # Download high-value only
                    if analysis['high_value_transactions'] > 0:
                        high_value_df = matches_df[matches_df['is_high_value'] == True]
                        high_value_csv = high_value_df.to_csv(index=False)
                        st.download_button(
                            label="💎 High-Value CSV",
                            data=high_value_csv,
                            file_name=f"high_value_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            width='stretch'
                        )
                    else:
                        st.info("No high-value transactions (>₹10K)")
                
                with col3:
                    # Download combined metadata
                    combined_metadata = {
                        entity1_name: st.session_state.entity1_data.get('metadata', {}),
                        entity2_name: st.session_state.entity2_data.get('metadata', {}),
                        'match_analysis': {
                            'total_transactions': analysis['total_transactions'],
                            'total_value': analysis['total_value'],
                            'high_value_transactions': analysis['high_value_transactions'],
                            'first_transaction_date': analysis['first_transaction_date'],
                            'last_transaction_date': analysis['last_transaction_date'],
                            'average_transaction_value': analysis['average_transaction_value']
                        }
                    }
                    metadata_json = json.dumps(combined_metadata, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📋 Combined Metadata",
                        data=metadata_json,
                        file_name=f"combined_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        width='stretch'
                    )
        
        elif st.session_state.entity1_data and st.session_state.entity2_data:
            st.info("👆 Click 'Find Matching Transactions' in the sidebar to analyze the relationship between the two entities.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to the Multi-Entity Transaction Analyzer
        
        This tool helps you analyze transaction relationships between two entities (companies, individuals, or accounts).
        
        ### 🎯 How to Use (3 Simple Steps):
        
        **Step 1:** Enter names for your two entities in the sidebar
        
        **Step 2:** Upload bank statements (PDF or CSV format) for both entities
        
        **Step 3:** Click "Find Matching Transactions" to see the analysis
        
        ### 📊 What You'll Get:
        
        - **Simple Charts** - Easy-to-understand visualizations showing money flow
        - **Transaction Timeline** - See when transactions happened
        - **Amount Distribution** - Understand transaction patterns
        - **Monthly Summary** - View activity by month
        - **CSV Downloads** - Export all data for further analysis
        
        ### 🏦 Supported Formats:
        
        - **PDF**: Bank statements from 4500+ banks (automatic extraction)
        - **CSV**: Pre-formatted transaction data
        
        ### 🚀 Get Started:
        
        Upload your first statement using the sidebar →
        """)

if __name__ == "__main__":
    main()