from pyspark.sql import functions as F


def engineer_temporal_features(df):
    return df.withColumn("Hour", F.hour("Start_Time")) \
             .withColumn("Day_of_Week", F.dayofweek("Start_Time") - 2) \
             .withColumn("Month", F.month("Start_Time")) \
             .withColumn("Year", F.year("Start_Time")) \
             .withColumn(
                 "Season",
                 F.when(F.col("Month").isin(12, 1, 2), "Winter")
                  .when(F.col("Month").isin(3, 4, 5), "Spring")
                  .when(F.col("Month").isin(6, 7, 8), "Summer")
                  .otherwise("Fall")
             )

