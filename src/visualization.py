from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def get_summary_statistics(df: DataFrame):
    numeric_summary = df.describe()
    print("Summary statistics for numeric columns:")
    numeric_summary.show()

    non_null_count = df.select([F.count(F.col(c)).alias(c) for c in df.columns])
    print("Non-null value count per column:")
    non_null_count.show()

    missing_values = df.select([
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)) for c in df.columns
    ])
    print("Missing values count per column:")
    missing_values.show()

    total_rows = df.count()
    missing_percentage = df.select([ 
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / total_rows * 100).alias(c)
        for c in df.columns
    ])
    print("Percentage of missing values per column:")
    missing_percentage.show()

    return numeric_summary, non_null_count, missing_values, missing_percentage
