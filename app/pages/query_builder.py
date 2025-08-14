import streamlit as st

def query_builder_page(spark):
    st.header("Query Builder")

    if 'spark_df' not in st.session_state or st.session_state.spark_df is None:
        st.warning("Please upload data first!")
        return

    df = st.session_state.spark_df
    df.createOrReplaceTempView("transactions")

    # --- Init session state once ---
    if "query" not in st.session_state:
        st.session_state.query = "SELECT * FROM transactions LIMIT 10"

    st.subheader("Interactive Query Builder")
    col1, col2 = st.columns([1, 3])

    # --- Example queries defined early so callbacks can use them ---
    examples = {
        "Fraud by Category": """
        SELECT 
            merchant_category, 
            COUNT(*) AS total,
            SUM(is_fraud) AS fraud_count,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_percentage
        FROM transactions
        GROUP BY merchant_category
        ORDER BY fraud_percentage DESC
        """,
        "Recent High-Value Transactions": """
        SELECT * FROM transactions
        WHERE amount > 1000
        ORDER BY timestamp DESC
        LIMIT 50
        """
    }

    # --- Callback updates session_state BEFORE the text_area is built on rerun ---
    def load_example_callback():
        choice = st.session_state.get("example_choice")
        if choice:
            st.session_state.query = examples[choice].strip()

    with col2:
        st.subheader("Example Queries")
        st.selectbox(
            "Load example query",
            list(examples.keys()),
            key="example_choice"
        )
        st.button("Load Example", on_click=load_example_callback)

        # Now build the text area; it will read the (possibly updated) value
        query = st.text_area(
            "Enter your SQL query",
            key="query",
            height=200,
        )

        if st.button("Run Query", key="run_query_btn"):
            try:
                result = spark.sql(query)
                pdf = result.limit(1000).toPandas()
                st.dataframe(pdf)
                st.download_button(
                    label="Download results as CSV",
                    data=pdf.to_csv(index=False).encode("utf-8"),
                    file_name='query_results.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Query error: {str(e)}")

    with col1:
        st.markdown("**Available Columns**")
        for c in df.columns:
            st.code(c)
