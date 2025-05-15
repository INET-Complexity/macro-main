"""
H5 Data Analysis Dashboard

This script creates a Streamlit dashboard to visualize H5 output data from the model.
It should be run from a directory containing H5 output files.

Requirements:
- streamlit
- pandas
- h5py
- plotly
- numpy

Run with (from inet-macro-dev directory):
    streamlit run path/to/macromodel/util/dash_h5.py
"""

import streamlit as st
import pandas as pd
import h5py
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Macromodel Dashboard - H5 output",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("Macromodel Dashboard - H5 output")

# Find all H5 files in the working directory and its subdirectories
cwd = os.getcwd()
h5_files = list(Path(cwd).glob("**/*.h5"))
if not h5_files:
    st.error("No H5 files found in the working directory")
    st.error("Please ensure you are running this script from a directory containing H5 output files")
    st.stop()

# File selection
selected_file = st.sidebar.selectbox("Select H5 File", h5_files)

# Function to load H5 data
def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # Get all datasets
        datasets = {}
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[:]
        f.visititems(collect_datasets)
    return datasets

# Function to parse dataset names into components
def parse_dataset_name(name):
    parts = name.split('/')
    if len(parts) >= 3:
        country = parts[0]
        agent_market = parts[1]
        variable = '/'.join(parts[2:])
        return country, agent_market, variable
    return None, None, name

# Function to get available options based on selections
def get_available_options(data, selected_country=None, selected_agent_market=None):
    available_agent_markets = set()
    available_variables = set()
    
    for name in data.keys():
        country, agent_market, variable = parse_dataset_name(name)
        if country and (selected_country is None or country == selected_country):
            if agent_market:
                available_agent_markets.add(agent_market)
                if selected_agent_market is None or agent_market == selected_agent_market:
                    if variable:
                        available_variables.add(variable)
    
    return sorted(list(available_agent_markets)), sorted(list(available_variables))

# Function to create fan chart
def create_fan_chart(df, title):
    # Calculate statistics
    mean = df.mean()
    q1 = df.quantile(0.25)  # First quartile
    q3 = df.quantile(0.75)  # Third quartile
    d1 = df.quantile(0.1)   # First decile
    d9 = df.quantile(0.9)   # Ninth decile
    
    # Create time points
    time_points = np.arange(len(df))
    
    # Create the fan chart
    fig = go.Figure()
    
    # Add the mean line
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean,
        name='Mean',
        line=dict(color='black', width=2)
    ))
    
    # Add deciles (from outer to inner)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=d9,
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='90th Percentile',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=d1,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='10th-90th Percentile'
    ))
    
    # Add quartiles (from outer to inner)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=q3,
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.4)',
        name='75th Percentile',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=q1,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.4)',
        name='25th-75th Percentile'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Load data
try:
    data = load_h5_data(selected_file)
    
    # Sidebar for dataset selection
    st.sidebar.header("Variable Selection")
    
    # Get initial lists of all components
    countries = sorted(list(set(parse_dataset_name(name)[0] for name in data.keys() if parse_dataset_name(name)[0])))
    
    # Create vertically stacked dropdowns in the sidebar
    selected_country = st.sidebar.selectbox("Country", countries)
    
    # Get available agent/markets based on selected country
    available_agent_markets, _ = get_available_options(data, selected_country)
    
    selected_agent_market = st.sidebar.selectbox("Agent/Market", available_agent_markets)
    
    # Get available variables based on selected country and agent/market
    _, available_variables = get_available_options(data, selected_country, selected_agent_market)
    
    selected_variable = st.sidebar.selectbox("Variable", available_variables)
    
    # Construct the full dataset name
    selected_dataset = f"{selected_country}/{selected_agent_market}/{selected_variable}"
    
    # Main content
    st.header(f"Variable: {selected_dataset}")
    
    # Convert selected dataset to DataFrame if possible
    try:
        df = pd.DataFrame(data[selected_dataset])
        df_t = df.T
        
        # Display data shape
        st.write(f"Shape: {data[selected_dataset].shape}")
        
        # Create visualizations based on data shape
        if len(data[selected_dataset].shape) == 2 and data[selected_dataset].shape[1] > 1000:
            # If second dimension is large, show fan chart
            # Transpose DataFrame for fan chart
            st.subheader("Descriptive Statistics")
            st.write(df_t.describe())
            fig = create_fan_chart(df_t, f"Fan Chart of {selected_dataset}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Otherwise show time series
            # Only show descriptive statistics if there are multiple observations
            if data[selected_dataset].shape[1] > 1:
                st.subheader("Descriptive Statistics")
                st.write(df_t.describe())
            fig = px.line(df, title=f"Time Series of {selected_dataset}")
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not convert dataset to DataFrame: {str(e)}")
        st.write("Raw data shape:", data[selected_dataset].shape)
        st.write("Raw data sample:", data[selected_dataset][:5])

except Exception as e:
    st.error(f"Error loading H5 file: {str(e)}")