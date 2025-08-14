from pyspark.sql.functions import *
from pyspark.sql.types import *

def add_time_features(df):
    return df \
        .withColumn("hour_of_day", hour(col("timestamp"))) \
        .withColumn("day_of_week", dayofweek(col("timestamp"))) \
        .withColumn("month", month(col("timestamp")))

def add_distance_feature(df):
    # First check if home location columns exist
    if 'home_lat' in df.columns and 'home_long' in df.columns:
        return df.withColumn("distance_from_home", 
            sqrt(pow(col("location_lat") - col("home_lat"), 2) + 
                 pow(col("location_long") - col("home_long"), 2)))
    else:
        # If home location not available, calculate distance from (0,0) as fallback
        # Or you could return the original DF without this feature
        return df.withColumn("distance_from_home", 
            sqrt(pow(col("location_lat"), 2) + 
                 pow(col("location_long"), 2)))

def add_velocity_feature(df):
    return df.withColumn(
        "txn_velocity",
        when(col("time_since_last").isNull() | (col("time_since_last") == 0), lit(None))
         .otherwise(lit(1.0) / col("time_since_last"))
    )

def is_off_hours(hour):
    return 1 if hour < 6 or hour > 22 else 0

off_hours_udf = udf(is_off_hours, IntegerType())

def add_off_hours_feature(df):
    return df.withColumn(
        "off_hours",
        when(
            col("hour_of_day").isNull(), lit(0)
        ).when(
            (col("hour_of_day") < 6) | (col("hour_of_day") > 22), lit(1)
        ).otherwise(lit(0)).cast(IntegerType())
    )

def prepare_features(df):
    df = add_time_features(df)
    df = add_distance_feature(df)
    df = add_velocity_feature(df)
    df = add_off_hours_feature(df)
    return df