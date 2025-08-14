import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.data_processing import load_data, clean_data
except ImportError:
    # Alternative import path for development
    from src.data_processing import load_data, clean_data

import streamlit as st
from src.data_processing import load_data, clean_data
from src.feature_engineering import prepare_features
import os

# Import page functions
from app.pages.data_exploration import data_exploration_page
from app.pages.query_builder import query_builder_page
from app.pages.ml_module import ml_module_page
from app.pages.results import results_page

# Initialize data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Configure page
st.set_page_config(
    page_title="Finance Fraud Analyzer",
    layout="wide"
)

# Initialize Spark session (cached)
@st.cache_resource
def init_spark():
    from src.data_processing import create_spark_session
    return create_spark_session()

def main():
    st.title("Finance Fraud Analyzer")
    st.markdown("""
    A comprehensive tool for detecting fraudulent transactions using PySpark.
    """)
    
    # Initialize session state
    if 'spark_df' not in st.session_state:
        st.session_state.spark_df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Data Upload",
        "Data Exploration",
        "Query Builder",
        "ML Module",
        "Results Dashboard"
    ])
    
    # Initialize Spark
    spark = init_spark()
    
    # Page routing
    if page == "Data Upload":
        data_upload_page(spark)
    elif page == "Data Exploration":
        data_exploration_page(spark)
    elif page == "Query Builder":
        query_builder_page(spark)
    elif page == "ML Module":
        ml_module_page(spark)
    elif page == "Results Dashboard":
        results_page(spark)

def data_upload_page(spark):
    st.header("Data Upload Module")

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload New Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Save to temp file
            temp_path = os.path.join("data", "temp_upload.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load data
            df = load_data(spark, temp_path)
            st.session_state.spark_df = df
            st.success("Data uploaded successfully!")
    
    with col2:
        st.subheader("Use Existing Dataset")
        existing_files = [f for f in os.listdir("data") if f.endswith('.csv')]
        
        if existing_files:
            selected_file = st.selectbox(
                "Select dataset",
                existing_files,
                index=0
            )
            
            if st.button("Load Selected Dataset"):
                df = load_data(spark, os.path.join("data", selected_file))
                st.session_state.spark_df = df
                st.success(f"Loaded {selected_file} successfully!")
        else:
            st.warning("No existing datasets found in data/ directory")

    # Show basic info if data is loaded
    if st.session_state.spark_df is not None:
        st.subheader("Dataset Info")
        st.write(f"Number of rows: {st.session_state.spark_df.count()}")
        st.write(f"Number of columns: {len(st.session_state.spark_df.columns)}")
        
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                st.session_state.cleaned_df = clean_data(st.session_state.spark_df)
                st.session_state.features_df = prepare_features(st.session_state.cleaned_df)
                st.success("Data cleaning and feature engineering complete!")

# Other page functions would go here...

if __name__ == "__main__":
    main()