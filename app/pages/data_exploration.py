import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_processing import explore_data
from pyspark.sql.types import NumericType
from pyspark.sql.functions import mean, stddev, min, max

def data_exploration_page(spark):
    st.header("Data Exploration")
    
    if 'spark_df' not in st.session_state or st.session_state.spark_df is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.spark_df
    
    # Show data preview
    st.subheader("Data Preview")
    preview_size = st.slider("Number of rows to preview", 5, 100, 10)
    pandas_df = df.limit(preview_size).toPandas()
    st.dataframe(pandas_df)
    
    # Basic statistics
    st.subheader("Basic Statistics")
    if st.button("Show Statistics"):
        stats_df = df.describe().toPandas()
        st.dataframe(stats_df)
    
    # Column explorer - numeric columns only
    st.subheader("Column Explorer (Numeric Columns Only)")
    
    # Get only numeric columns
    numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]
    
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset!")
        return
    
    selected_col = st.selectbox("Select column", numeric_cols)
    
    # Create columns with custom width ratio
    stats_col, viz_col = st.columns([3, 7])  # 30%/70% split
    
    with stats_col:
        st.write("**Column Statistics**")
        st.write(f"- **Unique values:** {df.select(selected_col).distinct().count()}")
        st.write(f"- **Missing values:** {df.filter(df[selected_col].isNull()).count()}")
        st.write(f"- **Data type:** {str(df.schema[selected_col].dataType)}")
        
        # Basic stats
        stats = df.select(
            mean(selected_col).alias('mean'),
            stddev(selected_col).alias('stddev'),
            min(selected_col).alias('min'),
            max(selected_col).alias('max')
        ).first()
        
        st.write(f"- **Mean:** {stats['mean']:.2f}")
        st.write(f"- **Std Dev:** {stats['stddev']:.2f}")
        st.write(f"- **Min:** {stats['min']}")
        st.write(f"- **Max:** {stats['max']}")
    
    with viz_col:
        st.write("**Visualization**")
        # Create smaller figure
        fig, ax = plt.subplots(figsize=(6, 3))  # Smaller plot size
        
        # Convert to pandas for visualization
        plot_data = df.select(selected_col).toPandas()
        
        # Create appropriate plot based on cardinality
        unique_count = df.select(selected_col).distinct().count()
        if unique_count > 10:
            plot_data[selected_col].plot.hist(ax=ax, bins=20)
            ax.set_title("Distribution")
            ax.set_xlabel(selected_col)
        else:
            value_counts = df.groupBy(selected_col).count().orderBy("count", ascending=False).toPandas()
            sns.barplot(x=selected_col, y="count", data=value_counts, ax=ax)
            ax.set_title("Value Counts")
            plt.xticks(rotation=45)
        
        st.pyplot(fig, use_container_width=True)  # Use container width
    
    
    
    # Automated exploration
    st.subheader("Automated Exploration")
    if st.button("Run Full Exploration"):
        with st.spinner("Analyzing data..."):
            results = explore_data(df)
            
            st.success("Exploration complete!")
            
            # Display results
            st.write("### Fraud Statistics")
            st.dataframe(results["fraud_stats"])
            
            st.write("### Top Risky Merchant Categories")
            st.dataframe(results["merchant_stats"])
            
            st.write("### Fraud by Device Type")
            st.dataframe(results["device_stats"])

if __name__ == "__main__":
    from app.streamlit_app import init_spark
    data_exploration_page(init_spark())