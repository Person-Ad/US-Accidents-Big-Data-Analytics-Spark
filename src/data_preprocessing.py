from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType, DoubleType, TimestampType, IntegerType
import pyspark.sql.functions as F
from typing import List, Optional

# 1. Remove irrelevant columns

def remove_irrelevant_columns(df: DataFrame) -> DataFrame:
    irrelevant = [
        "ID", "Source", "Description", "Street", "City", "County", "Zipcode",
        "Airport_Code", "Weather_Timestamp", "Wind_Direction",
        "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"
    ]
    return df.drop(*[c for c in irrelevant if c in df.columns])


# 2. Drop columns with high missing rates

def drop_high_missing_columns(df: DataFrame) -> DataFrame:
    to_drop = [c for c in ["End_Lat", "End_Lng", "Wind_Chill(F)", "Precipitation(in)"] if c in df.columns]
    return df.drop(*to_drop)


# 3. Remove columns with a single unique value

def remove_single_unique_value_columns(df: DataFrame) -> DataFrame:
    counts = df.agg(*[F.approx_count_distinct(c).alias(c) for c in df.columns]).first().asDict()
    single_val_cols = [c for c, cnt in counts.items() if cnt == 1]
    return df.drop(*single_val_cols)


# 4. Replace NaN with None

def replace_nan_with_none(
    df: DataFrame,
    columns: Optional[List[str]] = None,
    exclude_types: Optional[List[str]] = None
) -> DataFrame:
    exclude_types = exclude_types or ['boolean']
    col_types = dict(df.dtypes)
    if columns is None:
        columns = [c for c, t in col_types.items() if t not in exclude_types]
    else:
        invalid_cols = [c for c in columns if c not in col_types]
        if invalid_cols:
            raise ValueError(f"Columns {invalid_cols} not found in DataFrame")
    
    select_expr = []
    for c in df.columns:
        if c in columns:
            if col_types[c] in ['double', 'float']:
                select_expr.append(F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).alias(c))
            else:
                select_expr.append(F.when(F.col(c).isNull(), None).otherwise(F.col(c)).alias(c))
        else:
            select_expr.append(F.col(c))
    return df.select(*select_expr)



# 5. Impute missing values: numeric->mean, categorical->mode

def impute_missing_values(df: DataFrame) -> DataFrame:
    num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

    # numeric columns
    for c in num_cols:
        mean_val = df.select(F.mean(F.col(c))).first()[0]
        if mean_val is not None:
            df = df.fillna({c: mean_val})

    # categorical columns
    for c in str_cols:
        mode_row = df.groupBy(c).count().orderBy(F.desc("count")).first()
        if mode_row:
            df = df.fillna({c: mode_row[0]})

    return df


# 6. Cast columns to proper types

def cast_columns(df: DataFrame) -> DataFrame:
    num = [
        "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
        "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Distance(mi)"
    ]
    ts = ["Start_Time", "End_Time",]
    ints = ["Severity"]

    for c in num:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
    for c in ts:
        if c in df.columns:
            df = df.withColumn(c, F.to_timestamp(F.col(c)))
    for c in ints:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(IntegerType()))
    return df


# 7. Rename columns for units consistency

def rename_columns(df: DataFrame, mapping: dict) -> DataFrame:
    for old, new in mapping.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    return df


# 8. Convert distance miles->km

def convert_distance_km(df: DataFrame, col_name: str = "Distance") -> DataFrame:
    if col_name in df.columns:
        return df.withColumn(col_name, F.col(col_name) * 1.60934)
    return df


# 9. Convert and filter temperature (°F->°C, filter >threshold)

def convert_and_filter_temperature(
    df: DataFrame,
    col_name: str = "Temperature",
    max_c: float = 56.7
) -> DataFrame:
    if col_name in df.columns:
        df = df.withColumn(col_name, (F.col(col_name) - 32) * 5.0 / 9.0)
        df = df.filter(F.col(col_name) <= max_c)
    return df


# 10. Convert wind chill (°F->°C)

def convert_wind_chill(df: DataFrame, col_name: str = "Wind_Chill") -> DataFrame:
    if col_name in df.columns:
        return df.withColumn(col_name, (F.col(col_name) - 32) * 5.0 / 9.0)
    return df


# 11. Filter & convert wind speed (mph->m/s)

def filter_and_convert_wind_speed(
    df: DataFrame,
    col_name: str = "Wind_Speed",
    max_mph: float = 230.0
) -> DataFrame:
    if col_name in df.columns:
        df = df.filter(F.col(col_name) < max_mph)
        return df.withColumn(col_name, F.col(col_name) * 0.44704)
    return df


# 12. Filter & convert precipitation (in->mm)

def filter_and_convert_precipitation(
    df: DataFrame,
    col_name: str = "Precipitation",
    max_in: float = 20.0
) -> DataFrame:
    if col_name in df.columns:
        df = df.filter(F.col(col_name) < max_in)
        return df.withColumn(col_name, F.col(col_name) * 25.4)
    return df


# 13. Normalize wind direction

def normalize_wind_direction(df: DataFrame, col_name: str = "Wind_Direction") -> DataFrame:
    mapping = {
        'Calm': 'CALM', 'East': 'E', 'North': 'N',
        'South': 'S', 'West': 'W', 'Variable': 'VAR'
    }
    if col_name in df.columns:
        expr = F.col(col_name)
        for k, v in mapping.items():
            expr = F.when(F.col(col_name) == k, v).otherwise(expr)
        return df.withColumn(col_name, expr)
    return df


# 14. Merge weather conditions into main categories

def merge_weather_conditions(df: DataFrame, col_name: str = "Weather_Condition") -> DataFrame:
    if col_name in df.columns:
        return df.withColumn(
            col_name,
            F.when(F.col(col_name).rlike("(?i)fair|clear"), "Sunny")
             .when(F.col(col_name).rlike("(?i)rain|shower|drizzle|precipitation"), "Rainy")
             .when(F.col(col_name).rlike("(?i)fog|cloud|haze|dust|mist|overcast|smoke|ash"), "Cloudy")
             .when(F.col(col_name).rlike("(?i)snow|hail|sleet|ice|wintry"), "Snowy")
             .when(F.col(col_name).rlike("(?i)thunder|t-storm"), "Thunder")
             .when(F.col(col_name).rlike("(?i)storm|squalls"), "Storm")
             .when(F.col(col_name).rlike("(?i)tornado"), "Tornado")
             .otherwise("Other")
        )
    return df


# Master preprocessing pipeline
def preprocess_data(df: DataFrame) -> DataFrame:
    # 1. Drop irrelevant & high-missing columns
    df = remove_irrelevant_columns(df)
    df = drop_high_missing_columns(df)

    # 2. Remove constant features
    df = remove_single_unique_value_columns(df)

    # 3. Handle NaN (replace) and missing (impute)
    df = replace_nan_with_none(df)
    df = impute_missing_values(df)

    # 4. Cast types
    df = cast_columns(df)

    # 5. Rename for consistency
    rename_map = {
        'Distance(mi)': 'Distance',
        'Temperature(F)': 'Temperature',
        'Wind_Chill(F)': 'Wind_Chill',
        'Humidity(%)': 'Humidity',
        'Pressure(in)': 'Pressure',
        'Visibility(mi)': 'Visibility',
        'Wind_Speed(mph)': 'Wind_Speed',
        'Precipitation(in)': 'Precipitation'
    }
    df = rename_columns(df, rename_map)

    # 6. Unit conversions & filtering
    df = convert_distance_km(df)
    df = convert_and_filter_temperature(df)
    df = convert_wind_chill(df)
    df = filter_and_convert_wind_speed(df)
    df = filter_and_convert_precipitation(df)

    # 7. Normalize & merge categories
    df = normalize_wind_direction(df)
    df = merge_weather_conditions(df)

    return df
