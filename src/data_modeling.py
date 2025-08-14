from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when, lit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add class weights to handle imbalance with error handling
def add_class_weights(df, target_col="is_fraud"):
    
    try:
        # Count fraud vs non-fraud cases safely
        counts = df.groupBy(target_col).count().collect()
        if not counts:
            raise ValueError("No data available for class weight calculation")
            
        fraud_count = next((x.count for x in counts if x[target_col] == 1), 0)
        non_fraud_count = next((x.count for x in counts if x[target_col] == 0), 0)
        
        if fraud_count == 0 or non_fraud_count == 0:
            logger.warning("Extreme class imbalance detected - one class has zero instances")
            return df.withColumn("class_weight", lit(1.0))
        
        # Calculate weights
        total = fraud_count + non_fraud_count
        fraud_weight = non_fraud_count / total
        non_fraud_weight = fraud_count / total
        
        # Add weights column
        return df.withColumn(
            "class_weight",
            when(col(target_col) == 1, fraud_weight).otherwise(non_fraud_weight)
        )
    except Exception as e:
        logger.error(f"Error calculating class weights: {str(e)}")
        return df.withColumn("class_weight", lit(1.0))  # Fallback to no weights

# Build feature pipeline with type safety checks
def build_feature_pipeline(categorical_cols, numeric_cols):
    
    try:
        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
            for col in categorical_cols
        ]
        
        encoders = [
            OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded", handleInvalid="keep")
            for col in categorical_cols
        ]
        
        assembler = VectorAssembler(
            inputCols=[f"{col}_encoded" for col in categorical_cols] + numeric_cols,
            outputCol="features",
            handleInvalid="keep"
        )
        
        return Pipeline(stages=indexers + encoders + [assembler])
        
    except Exception as e:
        logger.error(f"Error building feature pipeline: {str(e)}")
        raise

#Train model with enhanced error handling and resource management
def train_model(train_df, model_type, categorical_cols, numeric_cols):
    try:
        # Add class weights with safe fallback
        weighted_df = add_class_weights(train_df).cache()
        
        # Build feature pipeline
        feature_pipeline = build_feature_pipeline(categorical_cols, numeric_cols)
        
        # Model configuration with additional safeguards
        if model_type == "Logistic Regression":
            model = LogisticRegression(
                featuresCol="features",
                labelCol="is_fraud",
                weightCol="class_weight",
                maxIter=50,  # Reduced for stability
                regParam=0.01,
                elasticNetParam=0.5,  # Added for regularization
                family="binomial",
                standardization=True
            )
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(
                featuresCol="features",
                labelCol="is_fraud",
                weightCol="class_weight",
                maxDepth=5,
                minInstancesPerNode=5,  # Added to prevent overfitting
                maxBins=32  # Reduced for memory efficiency
            )
        elif model_type == "Random Forest":
            model = RandomForestClassifier(
                featuresCol="features",
                labelCol="is_fraud",
                weightCol="class_weight",
                numTrees=20,  # Reduced for performance
                maxDepth=5,
                subsamplingRate=0.8  # Added for stability
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create and fit full pipeline with memory management
        pipeline = Pipeline(stages=feature_pipeline.getStages() + [model])
        
        # Fit model with reduced parallelism if needed
        spark = train_df.sparkSession
        original_parallelism = spark.conf.get("spark.sql.shuffle.partitions")
        try:
            if train_df.count() > 100000:  # Large dataset
                spark.conf.set("spark.sql.shuffle.partitions", "32")
                
            fitted_pipeline = pipeline.fit(weighted_df)
            return fitted_pipeline
            
        finally:
            spark.conf.set("spark.sql.shuffle.partitions", original_parallelism)
            weighted_df.unpersist()
            
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

# Evaluate model with additional metrics and safeguards
def evaluate_model(model, test_df):
    
    try:
        # Limit test data size if too large
        if test_df.count() > 10000:
            test_df = test_df.sample(0.1)
            
        predictions = model.transform(test_df).cache()
        
        # Calculate multiple metrics
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="is_fraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc = evaluator_auc.evaluate(predictions)
        
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="is_fraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )
        pr_auc = evaluator_pr.evaluate(predictions)
        
        # Calculate precision, recall, F1
        tp = predictions.filter("prediction = 1 AND is_fraud = 1").count()
        fp = predictions.filter("prediction = 1 AND is_fraud = 0").count()
        fn = predictions.filter("prediction = 0 AND is_fraud = 1").count()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate accuracy
        accuracy = predictions.filter("prediction = is_fraud").count() / predictions.count()
        
        metrics = {
            "AUC": auc,
            "PR_AUC": pr_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
            "Confusion Matrix": {
                "True Positives": tp,
                "False Positives": fp,
                "False Negatives": fn,
                "True Negatives": predictions.filter("prediction = 0 AND is_fraud = 0").count()
            }
        }
        
        return metrics, predictions
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        predictions.unpersist() if 'predictions' in locals() else None 