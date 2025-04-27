from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def add_complex_road_feature(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "Is_Complex_Road",
        F.when((F.col("Junction").cast("int") + F.col("Railway").cast("int") + F.col("Crossing").cast("int")) > 0, 1).otherwise(0)
    )

def engineer_temporal_features(df: DataFrame) -> DataFrame:
    df = df.withColumn("Hour", F.hour("Start_Time")) \
           .withColumn("Day_of_Week", F.dayofweek("Start_Time") - 2) \
           .withColumn("Month", F.month("Start_Time")) \
           .withColumn("Year", F.year("Start_Time")) \
           .withColumn(
               "Season",
               F.when(F.col("Month").isin(12, 1, 2), "Winter")
                .when(F.col("Month").isin(3, 4, 5), "Spring")
                .when(F.col("Month").isin(6, 7, 8), "Summer")
                .otherwise("Fall")
           ) \
           .withColumn("DayOfWeek", F.dayofweek("Start_Time")) \
           .withColumn("Duration", (F.unix_timestamp("End_Time") - F.unix_timestamp("Start_Time")) / 60)
    
    return df

def add_time_based_features(df: DataFrame) -> DataFrame:
    df = df.withColumn("Hour", F.hour("Start_Time"))
    df = df.withColumn("DayOfWeek", F.dayofweek("Start_Time"))
    df = df.withColumn("Month", F.month("Start_Time"))
    df = df.withColumn("Duration", (F.unix_timestamp("End_Time") - F.unix_timestamp("Start_Time")) / 60)
    return df

def add_weekend_feature(df: DataFrame) -> DataFrame:
    return df.withColumn("Is_Weekend", F.when(F.col("DayOfWeek").isin(1, 7), 1).otherwise(0))

def add_risk_score(df: DataFrame) -> DataFrame:
    return df.withColumn("Risk_Score", F.col("Severity") * F.col("Num_Accidents"))

def add_severe_and_high_score(df: DataFrame) -> DataFrame:
    df = df.withColumn("Is_Severe", F.when(F.col("Severity") >= 3, 1).otherwise(0))
    # df = df.withColumn("Is_High_Score", F.when(F.col("Risk_Score") >= 10, 1).otherwise(0))
    return df

def preprocess_features(df: DataFrame) -> DataFrame:
    df = add_complex_road_feature(df)
    df = engineer_temporal_features(df)
    df = add_time_based_features(df)
    df = add_weekend_feature(df)
    # df = add_risk_score(df)
    df = add_severe_and_high_score(df)
    
    return df
