"""
Professional Multi-Entity Transaction Analysis Dashboard
Enterprise-grade transaction analysis with pattern detection
FIXED VERSION - Currency symbols, charts, and visualizations improved
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import sys
import re

# ============ API KEY CONFIGURATION ============
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
if not GOOGLE_API_KEY:
    st.error("Google API Key not found! Please add GOOGLE_API_KEY to .streamlit/secrets.toml")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# ===============================================

from core.bank_statement_extractor import BankStatementExtractor
from core.transaction_matcher import TransactionMatcher

# Import pattern detector
try:
    from pattern_detector import PatternDetector
    from pattern_visualization import display_all_patterns
    PATTERN_DETECTION_AVAILABLE = True
except:
    PATTERN_DETECTION_AVAILABLE = False
    print("Pattern detector not found. Pattern detection features disabled.")

# Page configuration
st.set_page_config(
    page_title="Multi-Entity Transaction Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Optimized for both light and dark modes
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    
    /* Metric styling - works in both modes */
    .stMetric {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 15px;
        border-radius: 8px;
        color: white !important;
    }
    
    .stMetric label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Headers - adapt to theme */
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
        font-weight: 600;
    }
    
    /* Download buttons */
    .stDownloadButton button {
        width: 100%;
        background-color: #1e40af;
        color: white !important;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    
    .stDownloadButton button:hover {
        background-color: #1e3a8a;
    }
    
    /* Alert boxes - enhanced contrast */
    .critical-alert {
        background-color: #fee2e2;
        border-left: 5px solid #dc2626;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #7f1d1d !important;
    }
    
    .warning-alert {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #78350f !important;
    }
    
    .info-alert {
        background-color: #dbeafe;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #1e3a8a !important;
    }
    
    /* Rules card - good contrast */
    .rules-card {
        background-color: rgba(248, 250, 252, 0.8);
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .rules-card h4 {
        color: #1e40af !important;
        margin-bottom: 10px;
    }
    
    .rules-card ul {
        color: #334155;
    }
    
    /* Transaction blocks */
    .transaction-block {
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
    }
    
    .pattern-header {
        background-color: rgba(241, 245, 249, 0.9);
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-weight: 600;
        color: #1e293b !important;
    }
    
    /* Dataframe styling for better readability */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(248, 250, 252, 0.8);
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Text readability */
    .stMarkdown {
        color: inherit;
    }
    
    /* Ensure text in info boxes is readable */
    .stInfo, .stSuccess, .stWarning, .stError {
        color: #1e293b !important;
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
if 'patterns' not in st.session_state:
    st.session_state.patterns = None

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def classify_transaction_type(description: str) -> str:
    """Classify transaction based on description keywords"""
    if pd.isna(description):
        return 'Others'
    
    desc_upper = str(description).upper()
    
    if any(kw in desc_upper for kw in ['ATM', 'WITHDRAWAL', 'CASH', 'CDM', 'NWD', 'EAW']):
        return 'ATM/Cash Withdrawal'
    if any(kw in desc_upper for kw in ['NEFT', 'RTGS', 'FUND TRANSFER', 'FUND TRF']):
        return 'NEFT/RTGS'
    if any(kw in desc_upper for kw in ['UPI', 'UPIAR', 'UPIREF']):
        return 'UPI'
    if 'IMPS' in desc_upper:
        return 'IMPS'
    if any(kw in desc_upper for kw in ['CHQ', 'CHEQUE', 'CHECK']):
        return 'Cheque'
    if any(kw in desc_upper for kw in ['POS', 'CARD', 'SWIPE']):
        return 'Card/POS'
    if any(kw in desc_upper for kw in ['SALARY', 'SAL']):
        return 'Salary'
    
    return 'Others'

def analyze_transaction_types(df: pd.DataFrame, transaction_type_col: str = 'debit') -> pd.DataFrame:
    """Analyze transactions by type"""
    filtered_df = df[df[transaction_type_col].notna() & (df[transaction_type_col] > 0)].copy()
    
    if len(filtered_df) == 0:
        return pd.DataFrame(columns=['Type', 'Count', 'Total Amount', 'Percentage'])
    
    filtered_df['transaction_type'] = filtered_df['description'].apply(classify_transaction_type)
    
    summary = filtered_df.groupby('transaction_type').agg({
        transaction_type_col: ['count', 'sum']
    }).reset_index()
    
    summary.columns = ['Type', 'Count', 'Total Amount']
    summary['Percentage'] = (summary['Total Amount'] / summary['Total Amount'].sum() * 100).round(2)
    summary = summary.sort_values('Total Amount', ascending=False)
    
    return summary

def create_transaction_type_pie_chart(df: pd.DataFrame, transaction_type: str, title: str):
    """Create IMPROVED pie chart for transaction types with better visibility"""
    type_col = 'debit' if transaction_type == 'Debit' else 'credit'
    summary = analyze_transaction_types(df, type_col)
    
    if len(summary) == 0:
        return None
    
    # Color palette
    colors = ['#1e40af', '#059669', '#dc2626', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    
    fig = go.Figure(data=[go.Pie(
        labels=summary['Type'],
        values=summary['Total Amount'],
        hole=0.4,
        marker=dict(
            colors=colors[:len(summary)], 
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12, color='#1e293b', weight='bold'),  # IMPROVED: Better text visibility
        hovertemplate='<b>%{label}</b><br>' +
                      'Amount: ₹%{value:,.0f}<br>' +  # FIXED: Proper rupee symbol
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial", weight='bold', color='#1e293b'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=12, color='#1e293b', weight='bold')  # IMPROVED: Larger, bolder legend
        ),
        height=400,  # IMPROVED: Increased height for better visibility
        margin=dict(l=20, r=180, t=60, b=20),  # IMPROVED: More space for legend
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        font=dict(color='#1e293b')
    )
    
    # Add center text showing total
    total_amount = summary['Total Amount'].sum()
    center_text = f"Total<br>₹{total_amount/1000:.0f}K" if total_amount < 1000000 else f"Total<br>₹{total_amount/100000:.1f}L"
    fig.add_annotation(
        text=center_text,
        x=0.5, y=0.5,
        font=dict(size=16, family="Arial", weight='bold', color='#1e293b'),  # IMPROVED: Larger center text
        showarrow=False
    )
    
    return fig

def process_statement(uploaded_file, entity_name):
    """Process uploaded bank statement"""
    try:
        current_dir = os.getcwd()
        temp_dir = os.path.join(current_dir, "temp")
        extracted_dir = os.path.join(current_dir, "extracted_data")
        
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(extracted_dir, exist_ok=True)
        
        # Handle CSV files
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            
            required_cols = ['date', 'description', 'debit', 'credit', 'balance']
            if not all(col in df.columns for col in required_cols):
                return {
                    'success': False,
                    'error': f"CSV must contain columns: {', '.join(required_cols)}"
                }
            
            if 'transaction_id' not in df.columns:
                df['transaction_id'] = None
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
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
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            extractor = BankStatementExtractor()
            result = extractor.extract(temp_path, output_dir=extracted_dir)
            
            transactions_df = pd.read_csv(result['transactions_file'])
            transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors='coerce')
            transactions_df = transactions_df.dropna(subset=['date'])
            
            metadata = {}
            if result['metadata_file'] and os.path.exists(result['metadata_file']):
                with open(result['metadata_file'], 'r') as f:
                    metadata = json.load(f)
            
            if not metadata.get('account_holder_name'):
                metadata['account_holder_name'] = entity_name
            
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
    """Display entity summary with transaction type breakdown"""
    with col:
        st.markdown(f"### {entity_name}")
        
        if entity_data is None:
            st.info("No data uploaded yet")
            return
        
        df = entity_data['transactions']
        metadata = entity_data.get('metadata', {})
        
<<<<<<< HEAD
        # Account info in clean format
=======
        # Account info
>>>>>>> 0c964c0 (Final Demo Cut)
        st.markdown("**Account Details**")
        account_holder = metadata.get('account_holder_name', 'N/A')
        account_number = metadata.get('account_number', 'N/A')
        bank_name = metadata.get('bank_name', 'N/A')
        branch = metadata.get('branch', 'N/A')
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.text(f"Account Holder: {account_holder}")
            st.text(f"Bank: {bank_name}")
        with info_col2:
            st.text(f"Account: {account_number}")
            if branch != 'N/A':
                st.text(f"Branch: {branch}")
        
        # Statement period
        st.markdown("---")
        st.markdown("**Statement Period**")
        
        period_from = metadata.get('statement_period_from', 'N/A')
        period_to = metadata.get('statement_period_to', 'N/A')
        opening_balance = metadata.get('opening_balance', 'N/A')
        closing_balance = metadata.get('closing_balance', 'N/A')
        currency = metadata.get('currency', 'INR')
        
        if period_from != 'N/A' and period_to != 'N/A':
            st.text(f"Period: {period_from} to {period_to}")
        
        balance_col1, balance_col2 = st.columns(2)
        with balance_col1:
            if opening_balance != 'N/A':
                try:
                    opening_val = float(opening_balance)
                    st.text(f"Opening: ₹{opening_val:,.2f}")  # FIXED: Proper rupee symbol
                except:
                    st.text(f"Opening: {opening_balance}")
        with balance_col2:
            if closing_balance != 'N/A':
                try:
                    closing_val = float(closing_balance)
                    st.text(f"Closing: ₹{closing_val:,.2f}")  # FIXED: Proper rupee symbol
                except:
                    st.text(f"Closing: {closing_balance}")
        
        st.markdown("---")
        
<<<<<<< HEAD
        # Simple key metrics
=======
        # Transaction summary
>>>>>>> 0c964c0 (Final Demo Cut)
        st.markdown("**Transaction Summary**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Transactions", len(df))
            st.metric("Total Credits", f"₹{df['credit'].sum():,.0f}")  # FIXED: Proper rupee symbol
        
        with col2:
            #try:
                #date_col = pd.to_datetime(df['date'])
                #min_date = date_col.min()
                #max_date = date_col.max()
                
                #if pd.notna(min_date) and pd.notna(max_date):
                    #days = (max_date - min_date).days
                    #st.metric("Period (days)", days)
<<<<<<< HEAD
               # else:
=======
                #else:
>>>>>>> 3faee81 (Emoji Removed From Dashboard)
                    #st.metric("Period (days)", "N/A")
            #except:
                #st.metric("Period (days)", "N/A")
            
            st.metric("Total Debits", f"₹{df['debit'].sum():,.0f}")  # FIXED: Proper rupee symbol
        
        st.markdown("---")
        
        # Transaction type breakdown with IMPROVED PIE CHARTS
        st.markdown("**Transaction Type Analysis**")
        
        # Create two columns for debit and credit pie charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            debit_chart = create_transaction_type_pie_chart(df, 'Debit', 'Debit Transactions')
            if debit_chart:
                st.plotly_chart(debit_chart, width="stretch")
            else:
                st.info("No debit transactions")
        
        with chart_col2:
            credit_chart = create_transaction_type_pie_chart(df, 'Credit', 'Credit Transactions')
            if credit_chart:
                st.plotly_chart(credit_chart, width="stretch")
            else:
                st.info("No credit transactions")
        
        # Summary statistics below charts - IMPROVED visibility
        debit_summary = analyze_transaction_types(df, 'debit')
        credit_summary = analyze_transaction_types(df, 'credit')
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            if len(debit_summary) > 0:
                st.markdown("**Top Debit Categories:**")
                for _, row in debit_summary.head(3).iterrows():
                    st.text(f"• {row['Type']}: ₹{row['Total Amount']:,.0f} ({row['Percentage']:.1f}%)")  # FIXED
        
        with summary_col2:
            if len(credit_summary) > 0:
                st.markdown("**Top Credit Categories:**")
                for _, row in credit_summary.head(3).iterrows():
                    st.text(f"• {row['Type']}: ₹{row['Total Amount']:,.0f} ({row['Percentage']:.1f}%)")  # FIXED
        
        # Download buttons
        st.markdown("---")
        st.markdown("**Download Options**")
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Transactions CSV",
                data=csv_data,
                file_name=f"{entity_name.replace(' ', '_')}_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch",
                key=f"download_csv_{entity_name}"
            )
        
        with dl_col2:
            metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download Metadata JSON",
                data=metadata_json,
                file_name=f"{entity_name.replace(' ', '_')}_metadata_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                width="stretch",
                key=f"download_metadata_{entity_name}"
            )

def create_money_flow_chart(matches_df, entity1_name, entity2_name):
    """Create IMPROVED bar chart showing money flow with amounts on BOTH bars"""
    if len(matches_df) == 0:
        return None
    
    sent_by_e1 = matches_df[matches_df[f'{entity1_name}_action'] == 'sent']['amount'].sum()
    received_by_e1 = matches_df[matches_df[f'{entity1_name}_action'] == 'received']['amount'].sum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f'{entity1_name} → {entity2_name}', f'{entity2_name} → {entity1_name}'],
        y=[sent_by_e1, received_by_e1],
        text=[f'₹{sent_by_e1:,.0f}', f'₹{received_by_e1:,.0f}'],  # FIXED: Both bars show amounts now
        textposition='outside',  # IMPROVED: Show text outside bars for better visibility
        textfont=dict(size=14, color='#1e293b', weight='bold'),  # IMPROVED: Better text styling
        marker=dict(
            color=['#1e40af', '#059669'],
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.2f}<extra></extra>'  # FIXED
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Money Flow Analysis: Transaction Direction",
            font=dict(size=18, family="Arial", color="#1e293b", weight='bold')
        ),
        xaxis=dict(title="", showgrid=False, tickfont=dict(size=13, color='#1e293b')),
        yaxis=dict(
            title="Amount (₹)", 
            showgrid=True, 
            gridcolor='rgba(0,0,0,0.1)', 
            tickformat=',.0f',
            titlefont=dict(color='#1e293b'),
            tickfont=dict(color='#1e293b')
        ),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=450,  # IMPROVED: Increased height
        font=dict(size=13, color='#1e293b'),
        showlegend=False
    )
    
    return fig

def create_timeline_scatter(matches_df):
    """Create scatter plot with improved styling"""
    if len(matches_df) == 0:
        return None
    
    timeline_df = matches_df.copy()
    timeline_df['transaction_date'] = pd.to_datetime(timeline_df['transaction_date'])
    timeline_df = timeline_df.sort_values('transaction_date')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timeline_df['transaction_date'],
        y=timeline_df['amount'],
        mode='markers',
        name='Transactions',
        marker=dict(
            size=12, 
            color='#1e40af',
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Amount:</b> ₹%{y:,.0f}<extra></extra>'  # FIXED
    ))
    
    fig.update_layout(
        title=dict(
            text="Transaction Timeline",
            font=dict(size=18, family="Arial", color="#1e293b", weight='bold')
        ),
        xaxis=dict(
            title="Date", 
            showgrid=True, 
            gridcolor='rgba(0,0,0,0.1)',
            titlefont=dict(color='#1e293b'),
            tickfont=dict(color='#1e293b')
        ),
        yaxis=dict(
            title="Amount (₹)", 
            showgrid=True, 
            gridcolor='rgba(0,0,0,0.1)', 
            tickformat=',.0f',
            titlefont=dict(color='#1e293b'),
            tickfont=dict(color='#1e293b')
        ),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=400,
        hovermode='closest',
        font=dict(color='#1e293b')
    )
    
    return fig

def create_improved_sankey_diagram(matches_df, entity1_name, entity2_name):
    """Create IMPROVED Sankey diagram showing money flow between entities"""
    if len(matches_df) == 0:
        return None
    
    sent_by_e1 = matches_df[matches_df[f'{entity1_name}_action'] == 'sent']['amount'].sum()
    sent_by_e2 = matches_df[matches_df[f'{entity2_name}_action'] == 'sent']['amount'].sum()
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="white", width=2),
            label=[entity1_name, entity2_name],
            color=["#1e40af", "#059669"],
            customdata=[f"₹{sent_by_e1:,.0f}", f"₹{sent_by_e2:,.0f}"],
            hovertemplate='<b>%{label}</b><br>Total Sent: %{customdata}<extra></extra>'
        ),
        link=dict(
            source=[0, 1],  # indices correspond to labels
            target=[1, 0],
            value=[sent_by_e1, sent_by_e2],
            color=["rgba(30, 64, 175, 0.4)", "rgba(5, 150, 105, 0.4)"],
            customdata=[f"₹{sent_by_e1:,.0f}", f"₹{sent_by_e2:,.0f}"],
            hovertemplate='%{source.label} → %{target.label}<br>Amount: %{customdata}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="Money Flow: Sankey Diagram",
            font=dict(size=18, family="Arial", color="#1e293b", weight='bold')
        ),
        font=dict(size=14, color='#1e293b'),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def display_rules_settings():
    """Display the rules and thresholds being used"""
    st.markdown("## Analysis Rules & Configuration")
    
    st.markdown("""
    <div class='rules-card'>
    <h4>Transaction Classification Rules</h4>
    <ul>
        <li><strong>High-Value Transaction:</strong> Amount ≥ ₹10,000</li>
        <li><strong>Very High-Value:</strong> Amount ≥ ₹1,00,000</li>
        <li><strong>Round Amount:</strong> Multiples of ₹10,000, ₹50,000, or ₹1,00,000</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='rules-card'>
    <h4>Pattern Detection Rules</h4>
    <ul>
        <li><strong>Chunking Pattern:</strong> 3+ similar transactions within 30 days (±10% variance)</li>
        <li><strong>Rapid Withdrawal:</strong> ≥80% of received amount withdrawn within 72 hours</li>
        <li><strong>Round Amount Flag:</strong> High-value transactions (≥₹10,000) in round numbers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='rules-card'>
    <h4>Transaction Matching Rules</h4>
    <ul>
        <li><strong>Date Tolerance:</strong> ±3 days</li>
        <li><strong>Amount Tolerance:</strong> ±1%</li>
        <li><strong>Primary Match:</strong> UTR/RRN/Reference ID</li>
        <li><strong>Secondary Match:</strong> Amount + Date + Description similarity</li>
        <li><strong>Minimum Confidence:</strong> 50%</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    st.title("Multi-Entity Transaction Analysis System")
    st.markdown('*Professional transaction analysis with automated pattern detection*')
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Entity Configuration")
        
        entity1_name = st.text_input("Entity 1 Name", value=st.session_state.entity1_name)
        entity2_name = st.text_input("Entity 2 Name", value=st.session_state.entity2_name)
        
        if entity1_name != st.session_state.entity1_name or entity2_name != st.session_state.entity2_name:
            st.session_state.entity1_name = entity1_name
            st.session_state.entity2_name = entity2_name
            st.rerun()
        
        st.markdown("---")
        st.markdown("## Upload Statements")
        
        # Entity 1
        st.markdown(f"### {entity1_name}")
        entity1_file = st.file_uploader(
            f"Upload statement for {entity1_name}",
            type=['pdf', 'csv'],
            key='entity1'
        )
        
        if entity1_file and st.button(f"Process {entity1_name}", key='process1'):
            with st.spinner(f"Processing {entity1_name}'s statement..."):
                result = process_statement(entity1_file, entity1_name)
                if result['success']:
                    st.session_state.entity1_data = result
                    st.success(f"{entity1_name}: {result['transaction_count']} transactions processed")
                else:
                    st.error(f"Error: {result['error']}")
        
        st.markdown("---")
        
        # Entity 2
        st.markdown(f"### {entity2_name}")
        entity2_file = st.file_uploader(
            f"Upload statement for {entity2_name}",
            type=['pdf', 'csv'],
            key='entity2'
        )
        
        if entity2_file and st.button(f"Process {entity2_name}", key='process2'):
            with st.spinner(f"Processing {entity2_name}'s statement..."):
                result = process_statement(entity2_file, entity2_name)
                if result['success']:
                    st.session_state.entity2_data = result
                    st.success(f"{entity2_name}: {result['transaction_count']} transactions processed")
                else:
                    st.error(f"Error: {result['error']}")
        
        st.markdown("---")
        
        # Match and analyze button
        if st.session_state.entity1_data and st.session_state.entity2_data:
            if st.button("Find Matching Transactions", type="primary", width="stretch"):
                with st.spinner("Finding matching transactions..."):
                    entity1_df = st.session_state.entity1_data['transactions'].copy()
                    entity2_df = st.session_state.entity2_data['transactions'].copy()
                    
                    entity1_df = entity1_df.dropna(subset=['date'])
                    entity2_df = entity2_df.dropna(subset=['date'])
                    
                    matcher = TransactionMatcher()
                    matches = matcher.match_transactions(
                        entity1_df, entity2_df,
                        entity1_name, entity2_name
                    )
                    st.session_state.matches_df = matches
                    st.session_state.analysis = matcher.analyze_matches(matches)
                    
                    # Pattern detection
                    if PATTERN_DETECTION_AVAILABLE and len(matches) > 0:
                        try:
                            detector = PatternDetector(
                                chunking_window_days=30,
                                rapid_withdrawal_hours=72,
                                min_chunk_count=3
                            )
                            patterns = detector.analyze_patterns(
                                matches, entity1_df, entity2_df,
                                entity1_name, entity2_name
                            )
                            st.session_state.patterns = patterns
                        except Exception as e:
                            print(f"Pattern detection error: {e}")
                            st.session_state.patterns = None
                    
                    if len(matches) == 0:
                        st.warning(f"No matching transactions found")
                    else:
                        st.success(f"Found {len(matches)} matching transactions")
                    
                    st.rerun()
        
        st.markdown("---")
        
        # Reset
        if st.button("Reset All", width="stretch"):
            st.session_state.entity1_data = None
            st.session_state.entity2_data = None
            st.session_state.matches_df = None
            st.session_state.analysis = None
            st.session_state.patterns = None
            st.rerun()
    
    # Main content
    if st.session_state.entity1_data or st.session_state.entity2_data:
        
        # Rules display (collapsible)
        with st.expander("View Analysis Rules & Settings", expanded=False):
            display_rules_settings()
        
        # Show entity summaries
        col1, col2 = st.columns(2)
        display_entity_summary(st.session_state.entity1_data, entity1_name, col1)
        display_entity_summary(st.session_state.entity2_data, entity2_name, col2)
        
        # Detailed metadata viewer
        if st.session_state.entity1_data or st.session_state.entity2_data:
            with st.expander("View Detailed Metadata", expanded=False):
                st.markdown("### Complete Metadata Information")
                
                meta_col1, meta_col2 = st.columns(2)
                
                with meta_col1:
                    if st.session_state.entity1_data:
                        st.markdown(f"#### {entity1_name}")
                        metadata1 = st.session_state.entity1_data.get('metadata', {})
                        if metadata1:
                            st.json(metadata1)
                        else:
                            st.info("No metadata available")
                
                with meta_col2:
                    if st.session_state.entity2_data:
                        st.markdown(f"#### {entity2_name}")
                        metadata2 = st.session_state.entity2_data.get('metadata', {})
                        if metadata2:
                            st.json(metadata2)
                        else:
                            st.info("No metadata available")
        
        st.markdown("---")
        
        # Show matching results
        if st.session_state.matches_df is not None:
            matches_df = st.session_state.matches_df
            analysis = st.session_state.analysis
            
            if len(matches_df) == 0:
                st.markdown("## No Matching Transactions Found")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**{entity1_name}**: {len(st.session_state.entity1_data['transactions'])} transactions")
                with col2:
                    st.info(f"**{entity2_name}**: {len(st.session_state.entity2_data['transactions'])} transactions")
            
            else:
                # Key metrics - FIXED currency symbols
                st.markdown("## Transaction Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Matches", analysis['total_transactions'])
                with col2:
                    st.metric("Total Value", f"₹{analysis['total_value']:,.0f}")  # FIXED
                with col3:
                    st.metric("Average Transaction", f"₹{analysis['average_transaction_value']:,.0f}")  # FIXED
                with col4:
                    st.metric("High-Value (>₹10K)", analysis['high_value_transactions'])  # FIXED
                
                st.markdown("---")
                
                # Pattern Detection Section
                if st.session_state.patterns and PATTERN_DETECTION_AVAILABLE:
                    st.markdown("## Rule-Based Transaction Analysis")
                    display_all_patterns(
                        st.session_state.patterns, 
                        matches_df, 
                        st.session_state.entity1_data,
                        st.session_state.entity2_data,
                        entity1_name, 
                        entity2_name
                    )
                    st.markdown("---")
                
                # Transaction Details Section
                st.markdown("### Transaction Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    show_high_value = st.checkbox("Show only high-value (>₹10K)", value=False)  # FIXED
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
                
                # Display table
                display_cols = ['transaction_date', 'amount', f'{entity1_name}_action', f'{entity1_name}_description', f'{entity2_name}_description']
                st.dataframe(
                    filtered_df[display_cols],
                    width="stretch",
                    column_config={
                        "transaction_date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),  # FIXED
                        f"{entity1_name}_action": st.column_config.TextColumn(f"{entity1_name} Action"),
                        f"{entity1_name}_description": st.column_config.TextColumn(f"{entity1_name} Description"),
                        f"{entity2_name}_description": st.column_config.TextColumn(f"{entity2_name} Description")
                    }
                )
                
                st.markdown("---")
                
                # Visualizations - IMPROVED
                st.markdown("## Visual Analysis")
                
                tab1, tab2, tab3 = st.tabs([
                    "Money Flow (Bar Chart)", 
                    "Timeline",
                    "Money Flow (Sankey)"  # IMPROVED: Replace arrow diagram with Sankey
                ])
                
                with tab1:
                    fig = create_money_flow_chart(matches_df, entity1_name, entity2_name)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                
                with tab2:
                    fig = create_timeline_scatter(matches_df)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                
                with tab3:
                    fig = create_improved_sankey_diagram(matches_df, entity1_name, entity2_name)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                        
                        sent_by_e1 = matches_df[matches_df[f'{entity1_name}_action'] == 'sent']['amount'].sum()
                        received_by_e1 = matches_df[matches_df[f'{entity1_name}_action'] == 'received']['amount'].sum()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class='info-alert'>
                            <strong>{entity1_name} → {entity2_name}</strong><br>
                            Sent: ₹{sent_by_e1:,.0f}
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class='info-alert'>
                            <strong>{entity2_name} → {entity1_name}</strong><br>
                            Sent: ₹{received_by_e1:,.0f}
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Download section - FIXED
                st.markdown("## Download Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_all = matches_df.to_csv(index=False)
                    st.download_button(
                        label="Download All Matches CSV",
                        data=csv_all,
                        file_name=f"matched_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                
                with col2:
                    if analysis['high_value_transactions'] > 0:
                        high_value_df = matches_df[matches_df['is_high_value'] == True]
                        high_value_csv = high_value_df.to_csv(index=False)
                        st.download_button(
                            label="Download High-Value CSV",
                            data=high_value_csv,
                            file_name=f"high_value_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                    else:
                        st.info("No high-value transactions")
                
                with col3:
                    if st.session_state.patterns and PATTERN_DETECTION_AVAILABLE:
                        try:
                            detector = PatternDetector()
                            detector.detected_patterns = st.session_state.patterns
                            report = detector.generate_pattern_report()
                            st.download_button(
                                label="Download Pattern Report",
                                data=report,
                                file_name=f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                width="stretch"
                            )
                        except:
                            combined_metadata = {
                                entity1_name: st.session_state.entity1_data.get('metadata', {}),
                                entity2_name: st.session_state.entity2_data.get('metadata', {}),
                            }
                            metadata_json = json.dumps(combined_metadata, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="Download Combined Metadata",
                                data=metadata_json,
                                file_name=f"combined_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                width="stretch"
                            )
                    else:
                        combined_metadata = {
                            entity1_name: st.session_state.entity1_data.get('metadata', {}),
                            entity2_name: st.session_state.entity2_data.get('metadata', {}),
                        }
                        metadata_json = json.dumps(combined_metadata, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="Download Combined Metadata",
                            data=metadata_json,
                            file_name=f"combined_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            width="stretch"
                        )
        
        elif st.session_state.entity1_data and st.session_state.entity2_data:
            st.info("Click 'Find Matching Transactions' in the sidebar to analyze the relationship between the two entities.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Multi-Entity Transaction Analysis System
        
        This professional system analyzes transaction relationships between two entities (companies, individuals, or accounts) with automated pattern detection.
        
        ### How to Use:
        
        **Step 1:** Enter names for your two entities in the sidebar
        
        **Step 2:** Upload bank statements (PDF or CSV format) for both entities
        
        **Step 3:** Click "Find Matching Transactions" to perform analysis
        
        ### Analysis Features:
        
        - **Transaction Matching** - Multi-level matching using reference IDs, amounts, and dates
        - **Pattern Detection** - Automated detection of:
          - Chunking patterns (splitting large transactions into smaller ones)
          - Rapid withdrawals (quick withdrawal after receiving funds)
          - Round amount patterns (suspiciously round transactions)
        - **Transaction Type Analysis** - Breakdown by payment methods (ATM, NEFT, UPI, etc.)
        - **Visual Analytics** - Charts and graphs for easy interpretation
        - **Export Options** - Download results in CSV format
        
        ### Supported Formats:
        
        - **PDF**: Bank statements from 4500+ banks (automatic extraction)
        - **CSV**: Pre-formatted transaction data
        
        ### Get Started:
        
        Upload your first statement using the sidebar
        """)

if __name__ == "__main__":
    main()
