import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def results_page(spark):
    st.header("Results Dashboard")
    
    if 'test_results' not in st.session_state:
        st.warning("Please train a model first!")
        return
    
    results = st.session_state.test_results.toPandas()
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Predictions", len(results))
        st.metric("Positive Predictions", results['prediction'].sum())
    
    with col2:
        fig, ax = plt.subplots()
        results['prediction'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    
    # Performance visualization
    st.subheader("Performance Metrics")
    
    from sklearn.metrics import confusion_matrix, roc_curve
    cm = confusion_matrix(results['is_fraud'], results['prediction'])
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(results['is_fraud'], results['probability'].apply(lambda x: x[1]))
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    st.pyplot(fig)
    
    # Download results
    st.subheader("Download Results")
    csv = results.to_csv(index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    from app.streamlit_app import init_spark
    results_page(init_spark())