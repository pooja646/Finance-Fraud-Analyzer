from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def create_spark_session():
    return SparkSession.builder \
        .appName("FraudDetection") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def load_data(spark, path):
    return spark.read.csv(path, header=True, inferSchema=True)

def clean_data(df):
    # Handle missing values
    df = df.fillna({'device_type': 'unknown'})
    
    # Convert timestamp
    df = df.withColumn('timestamp', to_timestamp('timestamp'))
    
    # Remove duplicates
    df = df.dropDuplicates()
    
    return df


def explore_data(df):
    
    # Create local Spark session if none exists
    spark = SparkSession.builder.getOrCreate()
    
    # Register temp view for SQL queries
    df.createOrReplaceTempView("transactions")
    
    # Basic fraud statistics
    fraud_stats = spark.sql("""
        SELECT
            is_fraud,
            COUNT(*) as count,
            AVG(amount) as avg_amount,
            STDDEV(amount) as std_amount,
            MIN(amount) as min_amount,
            MAX(amount) as max_amount
        FROM transactions
        GROUP BY is_fraud
    """).toPandas()
    
    # Merchant category analysis
    merchant_stats = spark.sql("""
        SELECT 
            merchant_category,
            COUNT(*) as total,
            SUM(is_fraud) as fraud_count,
            ROUND(SUM(is_fraud)/COUNT(*)*100, 2) as fraud_percent
        FROM transactions
        GROUP BY merchant_category
        ORDER BY fraud_percent DESC
        LIMIT 10
    """).toPandas()
    
    # Device type analysis
    device_stats = spark.sql("""
        SELECT 
            device_type,
            COUNT(*) as total,
            SUM(is_fraud) as fraud_count,
            ROUND(SUM(is_fraud)/COUNT(*)*100, 2) as fraud_percent
        FROM transactions
        GROUP BY device_type
        ORDER BY fraud_percent DESC
    """).toPandas()
    
    return {
        "fraud_stats": fraud_stats,
        "merchant_stats": merchant_stats,
        "device_stats": device_stats
    }