"""
GDP Analysis Dashboard

This script creates a Streamlit dashboard to visualize GDP components and measures.
It should be run from a directory containing an 'output' folder with 'GBR_gdp_debug_output.csv'.

Requirements:
- streamlit
- pandas
- plotly

Run with:
    streamlit run path/to/macromodel/util/dash_gdp.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page config
st.set_page_config(
    page_title="GDP Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("GDP Analysis Dashboard")

# Read the data
@st.cache_data
def load_data():
    # Get the current working directory (should be equivalent to inet-macro-dev)
    cwd = os.getcwd()
    output_path = os.path.join(cwd, "output", "GBR_gdp_debug_output.csv")
    
    if not os.path.exists(output_path):
        st.error(f"Could not find GDP data at {output_path}")
        st.error("Please ensure you are running this script from a directory containing an 'output' folder with 'GBR_gdp_debug_output.csv'")
        st.stop()
        
    df = pd.read_csv(output_path)
    # Remove the first column if it's unnamed
    if df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
    # Reset index to ensure proper alignment
    df = df.reset_index(drop=True)
    return df

df = load_data()

# Set fixed y-axis limits
y_min = 0
y_max = 1_500_000_000_000  # 1.5 trillion

# Function to create stacked area chart
def create_gdp_chart(df, components, title, gdp_column):
    fig = go.Figure()
    
    # Add GDP line using the specified column
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[gdp_column],
        name='Total GDP',
        line=dict(color='black', width=2)
    ))
    
    # Add components as stacked areas
    for component in components:
        if component in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[component],
                name=component.replace('_', ' ').title(),
                stackgroup='one',
                fill='tonexty'
            ))
        else:
            st.error(f"Column {component} not found in DataFrame")
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Value (£)",
        hovermode='x unified',
        showlegend=True,
        height=600,
        legend=dict(
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            orientation="h"
        ),
        yaxis=dict(
            range=[y_min, y_max],
            tickformat=",.0f"
        ),
        margin=dict(b=100)
    )
    
    return fig

# Create tabs for different GDP approaches
tab1, tab2, tab3 = st.tabs(["Output Approach", "Expenditure Approach", "Income Approach"])

# Output Approach
with tab1:
    output_components = [
        '+Total_Output',
        '-Total_Intermediate_Consumption',
        '+Taxes_on_Products',
        '-Taxes_on_Production',
        '+Total_Real_Rent_Paid',
        '+Total_Imputed_Rent_Paid'
    ]
    fig_output = create_gdp_chart(df, output_components, "GDP Output Approach Components", "GDP_Output")
    st.plotly_chart(fig_output, use_container_width=True)

# Expenditure Approach
with tab2:
    expenditure_components = [
        '+Household_Consumption',
        '+Government_Consumption',
        '+Changes_in_Inventories',
        '+Gross_Fixed_Capital_Formation',
        '+Exports',
        '-Imports'
    ]
    fig_expenditure = create_gdp_chart(df, expenditure_components, "GDP Expenditure Approach Components", "GDP_Expenditure")
    st.plotly_chart(fig_expenditure, use_container_width=True)

# Income Approach
with tab3:
    income_components = [
        '+Operating_Surplus',
        '+Wages',
        '+Rent_Received',
        '+Total_Imputed_Rent_Paid',
        '+Central_Government_Rent_Received',
        '+Central_Government_Rental_Taxes',
        '+Central_Government_Product_Taxes'
    ]
    fig_income = create_gdp_chart(df, income_components, "GDP Income Approach Components", "GDP_Income")
    st.plotly_chart(fig_income, use_container_width=True)

# Add GDP comparison table
st.markdown("### GDP Measures Comparison")
gdp_comparison = df[['GDP_Output', 'GDP_Expenditure', 'GDP_Income']].copy()
gdp_comparison.columns = ['Output Approach', 'Expenditure Approach', 'Income Approach']
gdp_comparison.index.name = 'Time Period'
gdp_comparison = gdp_comparison.style.format('£{:,.2f}')
st.dataframe(gdp_comparison, use_container_width=True)

# Add a horizontal line to separate the footer
st.markdown("---")

# Create footer with statistics
st.markdown("### GDP Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average GDP", f"£{df['GDP_Output'].mean():,.2f}")
with col2:
    st.metric("Maximum GDP", f"£{df['GDP_Output'].max():,.2f}")
with col3:
    st.metric("Minimum GDP", f"£{df['GDP_Output'].min():,.2f}")
with col4:
    if st.checkbox("Show Raw Data"):
        st.dataframe(df) 