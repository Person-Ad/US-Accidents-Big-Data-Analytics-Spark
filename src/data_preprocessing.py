from pyspark.sql import DataFrame
from typing import Optional, List
from pyspark.sql.types import NumericType, StringType, DoubleType, TimestampType, IntegerType
import pyspark.sql.functions as F

def cast_columns(df: DataFrame) -> DataFrame:
    numeric_cols = [
        "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
        "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Distance(mi)"
    ]
    timestamp_cols = ["Start_Time", "End_Time", "Weather_Timestamp"]
    int_cols = ["Severity"]

    for col_name in numeric_cols:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
    
    for col_name in timestamp_cols:
        df = df.withColumn(col_name, F.col(col_name).cast(TimestampType()))

    for col_name in int_cols:
        df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))

    return df

def check_missing_values(df):
    total_rows = df.count()
    missing_values_df = df.select([
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / total_rows * 100).alias(c)
        for c in df.columns
    ])
    missing_values_dict = missing_values_df.first().asDict()
    sorted_missing_values = sorted(missing_values_dict.items(), key=lambda x: x[1], reverse=True)
    print("Missing Values Percentage by Column (Sorted):")
    for column, percentage in sorted_missing_values:
        print(f"{column}: {percentage:.2f}%")
    return missing_values_df

def drop_high_missing_columns(df: DataFrame) -> DataFrame:
    return df.drop("End_Lat", "End_Lng", "Wind_Chill(F)", "Precipitation(in)")

def impute_missing_values(df: DataFrame) -> DataFrame:
    num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    cat_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

    for col_name in num_cols:
        if col_name in df.columns:
            mean_val = df.select(col_name).na.drop().agg({col_name: 'mean'}).first()[0]
            if mean_val is not None:
                df = df.fillna({col_name: mean_val})

    for col_name in cat_cols:
        if col_name in df.columns:
            mode_row = (
                df.select(col_name)
                .na.drop()
                .groupBy(col_name)
                .count()
                .orderBy("count", ascending=False)
                .first()
            )
            if mode_row:
                mode_val = mode_row[0]
                df = df.fillna({col_name: mode_val})

    return df

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
            select_expr.append(F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).alias(c))
        else:
            select_expr.append(F.col(c))
    return df.select(*select_expr)

def remove_single_unique_value_columns_optimized(df):
    distinct_counts = df.agg(
        *[F.approx_count_distinct(F.col(c)).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    
    unique_value_columns = [col_name for col_name, count in distinct_counts.items() if count == 1]
    print("Features with one unique value:", unique_value_columns)
    
    df_cleaned = df.drop(*unique_value_columns)
    
    return df_cleaned

def remove_irrelevant_columns(df):
    irrelevant_columns = [
    "ID", "Source", "Description", "Street", "City", "County", "Zipcode", 
    "Airport_Code", "Weather_Timestamp", "Wind_Direction", 
    "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"
    ]
    df = df.drop(*irrelevant_columns)
    return df

def preprocess_data(df: DataFrame) -> DataFrame:
    
    df = remove_irrelevant_columns(df)
   
    df = drop_high_missing_columns(df)
    
    df = remove_single_unique_value_columns_optimized(df)

    df = impute_missing_values(df)
    
    # output_path = "../../data/processed/"  

    # df.write.mode("overwrite").option("header", "true").csv(output_path)

    # print(f"DataFrame saved to CSV at: {output_path}")
    
    return df



# -------------------------------- Collect -----------------------------------

#     #dropping End_lat and End_Lng since it is missing around 50% of the time
#     df = df.drop(columns = ['End_Lat', 'End_Lng'])

# #for precipitation, wind_chill and wind_speed, i will input the null values with the mean of the values as
# #dropping the null rows will take out too much of the data


# df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].mean())
# df['Wind_Chill(F)']= df['Wind_Chill(F)'].fillna(df['Wind_Chill(F)'].mean())
# df['Wind_Speed(mph)']=df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].mean())

# #for the rest of the values, since it is around 2% of the value, dropping the rows with the null value would be best to prevent
# #data from being more skewed

# df = df.dropna(subset = ['Street', 'City', 'Zipcode', 'Timezone', 'Airport_Code','Description',
#        'Weather_Timestamp', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
#        'Visibility(mi)', 'Wind_Direction', 'Weather_Condition',
#        'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
#        'Astronomical_Twilight'])


# #transforming Dates to datetime
# df['Start_Time'] =pd.to_datetime(df['Start_Time'].str.split('.').str[0])
# df['End_Time'] =pd.to_datetime(df['End_Time'].str.split('.').str[0])


# #rename columns
# df.rename(columns={
#     'Distance(mi)':'Distance',
#     'Temperature(F)':'Temperature',
#     'Wind_Chill(F)':'Wind_Chill',
#     'Humidity(%)':'Humidity',
#     'Pressure(in)':'Pressure',
#     'Visibility(mi)':'Visibility',
#     'Wind_Speed(mph)':'Wind_Speed',
#     'Precipitation(in)':'Precipitation',
# } , inplace = True)


# #transforming distance to km
# df.rename(columns={
#     'Distance(mi)':'Distance'
# }, inplace=True)

# df.Distance = df.Distance * 1.60934



# #transforming temperature to celsius

# df.Temperature = (df.Temperature - 32) * 5 / 9


# #dropping accidents with more than 60 degree as the highest temperature ever recorded is only 53 degree celsius
# #while the extreme negative degree is well within expectation
# pd.set_option('display.max_columns', 200)
# df[df.Temperature > 50]
# sns.boxplot(df, x = 'Temperature')

# df = df[df['Temperature'] < 60]

# #transforming wind_chill to celsius
# #also wind temperature looks within possibility

# df.Wind_Chill = (df.Wind_Chill - 32) * 5 / 9
# sns.boxplot(df, x='Wind_Chill')


# df['Wind_Direction'].value_counts().sort_index()


# #some of the values are the same ie. VAR and Variable, North and N, West and W, East and E, South and S, CALM and Calm
# WDmap = {'Calm':'CALM',
#          'East':'E',
#          'North':'N',
#          'South':'S',
#          'West':'W',
#          'Variable':'VAR',
#         }
# df['Wind_Direction'] = df['Wind_Direction'].replace(WDmap)


# # maximum windspeed in US that is recorded is 231mph, so deleting all wind_speed rows with windspeed higher than 230
# # changing mph to mps
# #https://wmo.asu.edu/content/northern-hemisphere-highest-wind
# sns.boxplot(df, x = 'Wind_Speed')
# df = df[df['Wind_Speed']<231]
# df['Wind_Speed'] = df['Wind_Speed'] * 0.44704


# sns.boxplot(df, x = 'Precipitation')


# # going to make the decision to remove precipitation over 20in as they are really big outliers
# # also changing the unit from in to mm
# df = df[df['Precipitation'] < 20]
# df['Precipitation'] = df['Precipitation'] * 25.4

# #too many weather conditions. narrow to these main types, Sunny, Cloudy, Rainy, Snowy, Thunder, storm, Tornados
# #https://www.twinkl.com.sg/teaching-wiki/different-types-of-weather-conditions#:~:text=Different%20types%20of%20weather%20conditions,-There%20are%20many&text=The%20five%20main%20types%20of,of%20the%20global%20weather%20system.
# pd.set_option('display.max_row', 200)
# df['Weather_Condition'].value_counts()
# df['Sunny'] = df['Weather_Condition'].str.contains('fair|clear', case = False)
# df['Rainy'] = df['Weather_Condition'].str.contains('rain|shower|drizzle|precipitation', case = False)
# df['Cloudy'] = df['Weather_Condition'].str.contains('Fog|Cloud|Haze|dust|Mist|overcast|smoke|ash', case = False)
# df['Snowy'] = df['Weather_Condition'].str.contains('snow|hail|sleet|ice|wintry', case = False)
# df['Thunder'] = df['Weather_Condition'].str.contains('thunder|T-Storm', case = False)
# df['Storm'] = df['Weather_Condition'].str.contains('storm|T-Storm|Squalls', case = False)
# df['Tornado'] = df['Weather_Condition'].str.contains('Tornado', case = False)


# df.to_csv('US Accident(clean).csv')

"""

Columns we'll analyse:
 Exploratory Analysis and Visualization

City
Start time
Start lat and long
Temperature
Weather Condition

city_by_accident = df.City.value_counts()
city_by_accident


"""

# # Filter out rows with missing or invalid coordinates
# df_CA = df_CA.dropna(subset=['Start_Lat', 'Start_Lng'])
# df_CA = df_CA[(df_CA['Start_Lat'] >= -90) & (df_CA['Start_Lat'] <= 90)]
# df_CA = df_CA[(df_CA['Start_Lng'] >= -180) & (df_CA['Start_Lng'] <= 180)]