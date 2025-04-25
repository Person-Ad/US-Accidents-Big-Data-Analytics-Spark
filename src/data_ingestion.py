from pyspark.sql import SparkSession
from pyspark.sql.types import *


def init_spark():
    return SparkSession.builder.master('local[*]').appName("US_Accidents_Analysis").getOrCreate()

def load_data(spark, file_path="../data/US_Accidents_March23.csv"):
    schema = StructType([
        StructField("ID", StringType(), True),
        StructField("Source", StringType(), True),
        StructField("Severity", IntegerType(), True),
        StructField("Start_Time", TimestampType(), True),
        StructField("End_Time", TimestampType(), True),
        StructField("Start_Lat", DoubleType(), True),
        StructField("Start_Lng", DoubleType(), True),
        StructField("End_Lat", DoubleType(), True),
        StructField("End_Lng", DoubleType(), True),
        StructField("Distance(mi)", DoubleType(), True),
        StructField("Description", StringType(), True),
        StructField("Street", StringType(), True),
        StructField("City", StringType(), True),
        StructField("County", StringType(), True),
        StructField("State", StringType(), True),
        StructField("Zipcode", StringType(), True),
        StructField("Country", StringType(), True),
        StructField("Timezone", StringType(), True),
        StructField("Airport_Code", StringType(), True),
        StructField("Weather_Timestamp", TimestampType(), True),
        StructField("Temperature(F)", DoubleType(), True),
        StructField("Wind_Chill(F)", DoubleType(), True),
        StructField("Humidity(%)", DoubleType(), True),
        StructField("Pressure(in)", DoubleType(), True),
        StructField("Visibility(mi)", DoubleType(), True),
        StructField("Wind_Direction", StringType(), True),
        StructField("Wind_Speed(mph)", DoubleType(), True),
        StructField("Precipitation(in)", DoubleType(), True),
        StructField("Weather_Condition", StringType(), True),
        
        StructField("Amenity", BooleanType(), True),
        StructField("Bump", BooleanType(), True),
        StructField("Crossing", BooleanType(), True),
        StructField("Give_Way", BooleanType(), True),
        StructField("Junction", BooleanType(), True),
        StructField("No_Exit", BooleanType(), True),
        StructField("Railway", BooleanType(), True),
        StructField("Roundabout", BooleanType(), True),
        StructField("Station", BooleanType(), True),
        StructField("Stop", BooleanType(), True),
        StructField("Traffic_Calming", BooleanType(), True),
        StructField("Traffic_Signal", BooleanType(), True),
        StructField("Turning_Loop", BooleanType(), True),

        StructField("Sunrise_Sunset", StringType(), True),
        StructField("Civil_Twilight", StringType(), True),
        StructField("Nautical_Twilight", StringType(), True),
        StructField("Astronomical_Twilight", StringType(), True)
    ])

    return spark.read.option("header", "true").schema(schema).csv(file_path)

if __name__ == "__main__":
    spark = init_spark()
    df = load_data(spark)
    df.show(5)  