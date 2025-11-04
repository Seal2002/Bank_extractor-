# dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime
from core.bank_statement_extractor import BankStatementExtractor

# Page configuration
st.set_page_config(
    page_title="Bank Statement Extractor Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'extraction_complete' not in st.session_state:
    st.session_state.extraction_complete = False
if 'result_data' not in st.session_state:
    st.session_state.result_data = None

def format_currency(amount, currency="INR"):
    """Format currency values"""
    if amount is None:
        return "N/A"
    try:
        # Convert to float if it's a string
        amount = float(amount)
        return f"{currency} {amount:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def process_bank_statement(uploaded_file, provider_type="gemini"):
    """Process the uploaded bank statement"""
    try:
        # Create temporary directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        os.makedirs("extracted_data", exist_ok=True)
        
        # Save uploaded file temporarily
        temp_path = os.path.join("temp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize extractor
        extractor = BankStatementExtractor()
        
        # Extract data
        with st.spinner("Extracting data from bank statement..."):
            result = extractor.extract(temp_path, output_dir="extracted_data")
        
        # Load extracted data
        transactions_df = pd.read_csv(result['transactions_file'])
        with open(result['metadata_file'], 'r') as f:
            metadata = json.load(f)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {
            'success': True,
            'transactions': transactions_df,
            'metadata': metadata,
            'transaction_count': result['transaction_count'],
            'transactions_file': result['transactions_file'],
            'metadata_file': result['metadata_file']
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def display_account_info(metadata):
    """Display account information in a clean format"""
    st.markdown("### 📋 Account Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Account Holder**")
        st.info(metadata.get('account_holder_name', 'N/A'))
        st.markdown("**Bank Name**")
        st.info(metadata.get('bank_name', 'N/A'))
    
    with col2:
        st.markdown("**Account Number**")
        st.info(metadata.get('account_number', 'N/A'))
        st.markdown("**Branch**")
        st.info(metadata.get('branch', 'N/A'))
    
    with col3:
        st.markdown("**Statement Period**")
        period_from = metadata.get('statement_period_from', 'N/A')
        period_to = metadata.get('statement_period_to', 'N/A')
        st.info(f"{period_from} to {period_to}")
        st.markdown("**Currency**")
        st.info(metadata.get('currency', 'INR'))

def display_balance_summary(metadata):
    """Display balance summary"""
    st.markdown("### 💰 Balance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    currency = metadata.get('currency', 'INR')
    opening = metadata.get('opening_balance')
    closing = metadata.get('closing_balance')
    
    with col1:
        st.metric(
            label="Opening Balance",
            value=format_currency(opening, currency)
        )
    
    with col2:
        st.metric(
            label="Closing Balance",
            value=format_currency(closing, currency)
        )
    
    with col3:
        if opening is not None and closing is not None:
            try:
                opening_float = float(opening)
                closing_float = float(closing)
                difference = closing_float - opening_float
                st.metric(
                    label="Net Change",
                    value=format_currency(abs(difference), currency),
                    delta=f"{difference:+.2f}"
                )
            except (ValueError, TypeError):
                st.metric(label="Net Change", value="N/A")

def display_transaction_summary(transactions_df):
    """Display transaction statistics"""
    st.markdown("### 📊 Transaction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(transactions_df)
    total_debits = transactions_df['debit'].sum()
    total_credits = transactions_df['credit'].sum()
    
    # Count transactions with transaction IDs
    transactions_with_id = transactions_df['transaction_id'].notna().sum()
    
    with col1:
        st.metric("Total Transactions", total_transactions)
    
    with col2:
        st.metric("Total Debits", format_currency(total_debits))
    
    with col3:
        st.metric("Total Credits", format_currency(total_credits))
    
    with col4:
        st.metric("With Transaction ID", transactions_with_id)

def display_transactions_table(transactions_df):
    """Display transactions in an interactive table"""
    st.markdown("### 📝 Transaction Details")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transaction_type = st.selectbox(
            "Filter by Type",
            ["All", "Debits Only", "Credits Only"]
        )
    
    with col2:
        if 'date' in transactions_df.columns:
            date_range = st.date_input(
                "Filter by Date Range",
                value=[],
                key="date_filter"
            )
    
    with col3:
        search_term = st.text_input("Search Description", "")
    
    # Apply filters
    filtered_df = transactions_df.copy()
    
    if transaction_type == "Debits Only":
        filtered_df = filtered_df[filtered_df['debit'] > 0]
    elif transaction_type == "Credits Only":
        filtered_df = filtered_df[filtered_df['credit'] > 0]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['description'].str.contains(search_term, case=False, na=False)
        ]
    
    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "description": st.column_config.TextColumn("Description", width="large"),
            "debit": st.column_config.NumberColumn("Debit", format="%.2f"),
            "credit": st.column_config.NumberColumn("Credit", format="%.2f"),
            "balance": st.column_config.NumberColumn("Balance", format="%.2f"),
            "transaction_id": st.column_config.TextColumn("Transaction ID")
        }
    )
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Transactions (CSV)",
            data=csv,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download Transactions (JSON)",
            data=json_data,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 style="text-align: center;">🏦 Bank Statement Extractor Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📤 Upload Bank Statement")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a bank statement PDF file (max 200MB)"
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        provider_type = st.selectbox(
            "LLM Provider",
            ["gemini", "openai"],
            help="Select the AI provider for extraction"
        )
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
        This dashboard extracts transaction data and account information from bank statements.
        
        **Features:**
        - Multi-bank support (4500+ banks)
        - Automatic deduplication
        - Transaction ID detection
        - Metadata extraction
        - Export to CSV/JSON
        """)
        
        if st.button("🔄 Reset"):
            st.session_state.extraction_complete = False
            st.session_state.result_data = None
            st.rerun()
    
    # Main content
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        if st.button("🚀 Extract Data", type="primary"):
            result = process_bank_statement(uploaded_file, provider_type)
            
            if result['success']:
                st.session_state.extraction_complete = True
                st.session_state.result_data = result
                st.success("✅ Extraction completed successfully!")
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # Display results
    if st.session_state.extraction_complete and st.session_state.result_data:
        result = st.session_state.result_data
        
        # Display account information
        display_account_info(result['metadata'])
        
        st.markdown("---")
        
        # Display balance summary
        display_balance_summary(result['metadata'])
        
        st.markdown("---")
        
        # Display transaction summary
        display_transaction_summary(result['transactions'])
        
        st.markdown("---")
        
        # Display transactions table
        display_transactions_table(result['transactions'])
    
    else:
        # Welcome message
        st.markdown("""
        ## 👋 Welcome to the Bank Statement Extractor Dashboard
        
        Upload a bank statement PDF using the sidebar to get started.
        
        The dashboard will:
        1. **Extract Account Information** - Bank name, account number, holder name, etc.
        2. **Process Transactions** - All debits, credits, and balances
        3. **Detect Transaction IDs** - Automatically identify unique transaction identifiers
        4. **Remove Duplicates** - Intelligent deduplication based on transaction IDs
        5. **Provide Analytics** - Summary statistics and insights
        
        ### 🎯 Supported Banks
        
        This system works with **4500+ banks** including:
        - HDFC Bank
        - SBI (State Bank of India)
        - ICICI Bank
        - Axis Bank
        - Allahabad Bank
        - And many more...
        """)

if __name__ == "__main__":
    main()
