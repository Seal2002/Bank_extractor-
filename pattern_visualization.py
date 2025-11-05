import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================
# Chart Functions
# ==============================

def create_money_flow_chart(df):
    st.subheader("💸 Money Flow by Transaction Type")

    # Ensure column names are valid
    if 'Type' not in df.columns or 'Amount' not in df.columns:
        st.warning("Missing 'Type' or 'Amount' column in data.")
        return

    money_flow = df.groupby('Type')['Amount'].sum().reset_index()

    fig = px.bar(
        money_flow,
        x='Type',
        y='Amount',
        color='Type',
        title='Total Inflow vs Outflow',
        text_auto=True
    )

    # Updated layout syntax
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text='Transaction Type',
                font=dict(size=14, color='black')
            )
        ),
        yaxis=dict(
            title=dict(
                text='Amount (₹)',
                font=dict(size=14, color='black')
            )
        ),
        title=dict(
            font=dict(size=18, color='black'),
            x=0.5
        ),
        bargap=0.3
    )

    st.plotly_chart(fig, width='stretch')


def create_category_spending_chart(df):
    st.subheader("📊 Spending by Category")

    if 'Category' not in df.columns or 'Amount' not in df.columns:
        st.warning("Missing 'Category' or 'Amount' column in data.")
        return

    category_spending = df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)

    fig = px.pie(
        category_spending,
        names='Category',
        values='Amount',
        title='Spending Distribution by Category',
        hole=0.4
    )

    fig.update_traces(textinfo='percent+label', textfont_size=13)
    fig.update_layout(title=dict(x=0.5, font=dict(size=18, color='black')))

    st.plotly_chart(fig, width='stretch')


def create_monthly_trend_chart(df):
    st.subheader("📈 Monthly Transaction Trend")

    if 'Date' not in df.columns or 'Amount' not in df.columns:
        st.warning("Missing 'Date' or 'Amount' column in data.")
        return

    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M').astype(str)
    monthly_sum = df.groupby('Month')['Amount'].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sum['Month'],
        y=monthly_sum['Amount'],
        mode='lines+markers',
        name='Monthly Total',
        line=dict(width=3)
    ))

    fig.update_layout(
        title=dict(
            text='Monthly Transaction Trends',
            x=0.5,
            font=dict(size=18, color='black')
        ),
        xaxis=dict(
            title=dict(text='Month', font=dict(size=14, color='black'))
        ),
        yaxis=dict(
            title=dict(text='Total Amount (₹)', font=dict(size=14, color='black'))
        )
    )

    st.plotly_chart(fig, width='stretch')


def create_top_entities_chart(df, column_name):
    st.subheader(f"🏦 Top 10 {column_name}s by Transaction Value")

    if column_name not in df.columns or 'Amount' not in df.columns:
        st.warning(f"Missing '{column_name}' or 'Amount' column in data.")
        return

    top_entities = df.groupby(column_name)['Amount'].sum().nlargest(10).reset_index()

    fig = px.bar(
        top_entities,
        x='Amount',
        y=column_name,
        orientation='h',
        title=f'Top 10 {column_name}s by Transaction Value',
        text_auto=True,
        color='Amount'
    )

    fig.update_layout(
        xaxis=dict(title=dict(text='Amount (₹)', font=dict(size=14, color='black'))),
        yaxis=dict(title=dict(text=column_name, font=dict(size=14, color='black'))),
        title=dict(x=0.5, font=dict(size=18, color='black')),
        bargap=0.3
    )

    st.plotly_chart(fig, width='stretch')


# ==============================
# Streamlit Section
# ==============================
def visualize_patterns(df):
    st.header("📉 Transaction Pattern Visualizations")

    if df.empty:
        st.warning("No data available for visualization.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Money Flow", "Category Spending", "Monthly Trends", "Top Entities"
    ])

    with tab1:
        create_money_flow_chart(df)

    with tab2:
        create_category_spending_chart(df)

    with tab3:
        create_monthly_trend_chart(df)

    with tab4:
        selected_column = st.selectbox(
            "Select Column for Top Entities",
            [col for col in df.columns if df[col].dtype == 'object']
        )
        create_top_entities_chart(df, selected_column)
