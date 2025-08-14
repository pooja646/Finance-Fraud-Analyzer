import streamlit as st
import pandas as pd
from src.data_modeling import build_feature_pipeline, train_model, evaluate_model

def ml_module_page(spark):
    st.header("Machine Learning Module")
    
    if 'features_df' not in st.session_state or st.session_state.features_df is None:
        st.warning("Please upload and clean data first!")
        return
    
    # Cache the dataframe to avoid recomputation
    df = st.session_state.features_df.cache()
    
    # Add try-except blocks for better error handling
    try:
        
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "Select target column",
                df.columns,
                index=df.columns.index('is_fraud') if 'is_fraud' in df.columns else 0
            )
            
            test_size = st.slider("Test set size (%)", 10, 40, 20)
        
        with col2:
            available_features = [col for col in df.columns if col != target_col]
            selected_features = st.multiselect(
                "Select features",
                available_features,
                default=['amount', 'distance_from_home', 'hour_of_day'] if 'distance_from_home' in available_features else available_features[:3]
            )
        
        # Model selection
        model_type = st.selectbox(
            "Select model type",
            ["Logistic Regression", "Decision Tree", "Random Forest"]
        )

        # Add hyperparameter tuning options based on model type
        if model_type == "Logistic Regression":
            reg_param = st.slider("Regularization Parameter", 0.0, 1.0, 0.1)
            max_iter = st.slider("Maximum Iterations", 10, 1000, 100)
        elif model_type == "Decision Tree":
            max_depth = st.slider("Max Depth", 2, 30, 5)
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Split data
                    train, test = df.randomSplit([1 - test_size/100, test_size/100], seed=42)
                    
                    # Identify feature types
                    categorical_cols = [col for col in selected_features 
                                    if str(df.schema[col].dataType) == 'StringType']
                    numeric_cols = [col for col in selected_features 
                                if str(df.schema[col].dataType) != 'StringType']
                    
                    # Train model
                    model = train_model(
                        train_df=train,
                        model_type=model_type,
                        categorical_cols=categorical_cols,
                        numeric_cols=numeric_cols
                    )
                    
                    # Evaluate
                    metrics, predictions = evaluate_model(model, test)
                    
                    # Store results
                    st.session_state.model = model
                    st.session_state.test_results = predictions
                    st.session_state.metrics = metrics
                    
                    # Show success message
                    st.success(f"{model_type} trained successfully!")
                    
                    # Display all metrics in an organized way
                    st.subheader("Model Performance Metrics")
                    
                    # Create columns for metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("AUC Score", f"{metrics.get('AUC', 0):.4f}")
                        st.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
                        
                    with col2:
                        st.metric("PR AUC Score", f"{metrics.get('PR_AUC', 0):.4f}")
                        st.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
                        
                    with col3:
                        st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
                        st.metric("F1 Score", f"{metrics.get('F1', 0):.4f}")
                    
                    # Confusion matrix visualization
                    st.subheader("Confusion Matrix")
                    conf_matrix = metrics.get('Confusion Matrix', {})
                    conf_df = pd.DataFrame([
                        ["True Negative", conf_matrix.get('True Negatives', 0)],
                        ["False Positive", conf_matrix.get('False Positives', 0)],
                        ["False Negative", conf_matrix.get('False Negatives', 0)],
                        ["True Positive", conf_matrix.get('True Positives', 0)]
                    ], columns=["Type", "Count"])
                    
                    st.dataframe(conf_df)
                    
                    # Show predictions
                    st.subheader("Sample Predictions")
                    st.dataframe(predictions.select(target_col, "prediction", "probability", *selected_features[:3]).limit(100).toPandas())
                    
                    # Show feature importance for tree-based models
                    if model_type in ["Decision Tree", "Random Forest"]:
                        if hasattr(model.stages[-1], 'featureImportances'):
                            feature_importance = model.stages[-1].featureImportances
                            importance_df = pd.DataFrame({
                                'feature': numeric_cols + [f"{col}_encoded" for col in categorical_cols],
                                'importance': [feature_importance[i] for i in range(len(numeric_cols) + len(categorical_cols))]
                            }).sort_values('importance', ascending=False)
                            
                            st.subheader("Feature Importance")
                            st.bar_chart(importance_df.set_index('feature'))
                        
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.error("Try reducing the data size or selecting fewer features")
        
    finally:
        df.unpersist()  # Clean up memory
    
    # Show metrics if available (for when returning to the page)
    if 'metrics' in st.session_state:
        st.subheader("Previous Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AUC Score", f"{st.session_state.metrics.get('AUC', 0):.4f}")
            st.metric("Precision", f"{st.session_state.metrics.get('Precision', 0):.4f}")
            
        with col2:
            st.metric("PR AUC Score", f"{st.session_state.metrics.get('PR_AUC', 0):.4f}")
            st.metric("Recall", f"{st.session_state.metrics.get('Recall', 0):.4f}")
            
        with col3:
            st.metric("Accuracy", f"{st.session_state.metrics.get('Accuracy', 0):.4f}")
            st.metric("F1 Score", f"{st.session_state.metrics.get('F1', 0):.4f}")
        
        # Feature importance
        if hasattr(st.session_state.model.stages[-1], 'featureImportances'):
            st.subheader("Feature Importance")
            feature_importance = st.session_state.model.stages[-1].featureImportances
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': [feature_importance[i] for i in range(len(selected_features))]
            }).sort_values('importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('feature'))

if __name__ == "__main__":
    from app.streamlit_app import init_spark
    ml_module_page(init_spark())