import matplotlib.pyplot as plt
import seaborn as sns

def plot_temporal_frequencies(df):
    """
    Plot accident frequencies by hour, day of week, and month.
    
    Parameters:
    df (DataFrame): PySpark DataFrame with preprocessed accident data.
    """
    freq_hour = df.groupBy("Hour").count().toPandas()
    freq_day = df.groupBy("DayOfWeek").count().toPandas()
    freq_month = df.groupBy("Month").count().toPandas()

    # Plot by hour
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Hour", y="count", data=freq_hour)
    plt.title("Accidents by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Accidents")
    plt.savefig("../output/visualizations/accidents_by_hour.png")
    plt.close()

    # Plot by day of week
    plt.figure(figsize=(10, 6))
    sns.barplot(x="DayOfWeek", y="count", data=freq_day)
    plt.title("Accidents by Day of Week")
    plt.xlabel("Day of Week (1=Sunday, 7=Saturday)")
    plt.ylabel("Number of Accidents")
    plt.savefig("../output/visualizations/accidents_by_day.png")
    plt.close()

    # Plot by month
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Month", y="count", data=freq_month)
    plt.title("Accidents by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.savefig("../output/visualizations/accidents_by_month.png")
    plt.close()


def plot_state_accident_rates(df):
    """
    Plot normalized accident rates by state (accidents per capita).
    
    Parameters:
    df (DataFrame): PySpark DataFrame with preprocessed accident data.
    """
    # State populations (example data; replace with actual population data)
    state_populations = {
        "CA": 39538223, "TX": 29145505, "FL": 21538187, "NY": 20201249,  # Add more as needed
    }
    
    # Aggregate accidents by state
    state_counts = df.groupBy("State").count().toPandas()
    
    # Normalize by population
    state_counts["Population"] = state_counts["State"].map(state_populations)
    state_counts["Accidents_Per_Capita"] = state_counts["count"] / state_counts["Population"] * 100000  # Per 100,000 people
    
    # Sort and select top 10 states
    state_counts = state_counts.sort_values("Accidents_Per_Capita", ascending=False).head(10)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Accidents_Per_Capita", y="State", data=state_counts)
    plt.title("Top 10 States by Normalized Accident Rates (Per 100,000 People)")
    plt.xlabel("Accidents Per 100,000 People")
    plt.ylabel("State")
    plt.savefig("../output/visualizations/normalized_accident_rates_by_state.png")
    plt.close()

from pyspark.ml.clustering import KMeans

def cluster_regions_by_temperature(df, k=2):
    """
    Cluster regions by temperature and accident frequency using KMeans.
    
    Parameters:
    df (DataFrame): PySpark DataFrame with scaled features.
    k (int): Number of clusters (default=2 for warm vs. cold).
    
    Returns:
    DataFrame: Clustered DataFrame with predictions.
    """
    # Apply KMeans clustering
    kmeans = KMeans(k=k, featuresCol="scaled_features")
    model = kmeans.fit(df)
    clustered_df = model.transform(df)
    
    # Summarize clusters by temperature category
    cluster_summary = clustered_df.select("Temp_Category", "prediction").groupBy("Temp_Category", "prediction").count().toPandas()
    cluster_summary.to_csv("../output/visualizations/temperature_clusters.csv")
    
    # Plot cluster centers (temperature vs. accident frequency)
    centers = model.clusterCenters()
    plt.figure(figsize=(10, 6))
    for i, center in enumerate(centers):
        plt.scatter(i, center[0], label=f"Cluster {i} (Temp Component)")
    plt.title("Cluster Centers (Temperature Component)")
    plt.xlabel("Cluster")
    plt.ylabel("Scaled Temperature Value")
    plt.legend()
    plt.savefig("../output/visualizations/cluster_centers_temperature.png")
    plt.close()
    
    return clustered_df


def plot_accident_severity_distribution(df):
    """
    Plot the distribution of accident severity.
    
    Parameters:
    df (DataFrame): PySpark DataFrame with preprocessed accident data.
    """
    severity_dist = df.groupBy("Severity").count().toPandas()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Severity", y="count", data=severity_dist)
    plt.title("Distribution of Accident Severity")
    plt.xlabel("Severity Level")
    plt.ylabel("Number of Accidents")
    plt.savefig("../output/visualizations/severity_distribution.png")
    plt.close()


    
def plot_weather_impact(df):
    """
    Plot accident frequency by weather condition.
    
    Parameters:
    df (DataFrame): PySpark DataFrame with preprocessed accident data.
    """
    weather_counts = df.groupBy("Weather_Condition").count().toPandas()
    weather_counts = weather_counts.sort_values("count", ascending=False).head(10)  # Top 10 conditions
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="count", y="Weather_Condition", data=weather_counts)
    plt.title("Top 10 Weather Conditions by Accident Frequency")
    plt.xlabel("Number of Accidents")
    plt.ylabel("Weather Condition")
    plt.savefig("../output/visualizations/accidents_by_weather.png")
    plt.close()


def descriptive_analytics(df, spark, k=2):
    """
    Run all descriptive analytics and generate visualizations.
    
    Parameters:
    df (DataFrame): PySpark DataFrame with preprocessed and feature-engineered data.
    spark (SparkSession): Active Spark session.
    k (int): Number of clusters for KMeans.
    """
    # Temporal analysis
    # plot_temporal_frequencies(df)
    
    # State-wise normalized rates
    plot_state_accident_rates(df)
    
    # Clustering by temperature
    clustered_df = cluster_regions_by_temperature(df, k)
    
    # Severity distribution
    plot_accident_severity_distribution(df)
    
    # Weather impact
    plot_weather_impact(df)

if __name__ == "__main__":
    pass
    # spark = init_spark()
    # df = load_data(spark)
    # descriptive_analytics(df, spark)


"""
Key Insights:
Road features like 'Traffic_Signal' and 'Junction' significantly impact accident frequency and severity.

Most accidents occur under clear weather conditions, but adverse weather still plays a role.

Accidents peak during rush hours and are more frequent on weekdays.

Urban areas, particularly large cities, are hotspots for traffic accidents.

Lower visibility is associated with higher accident severity.

Surprising Patterns:
A significant number of accidents occur under clear weather conditions, which might be due to higher traffic volumes.

The correlation between visibility and accident severity highlights the importance of maintaining good visibility conditions on the road.

Recommendations:
Implementing stricter traffic control measures at junctions and traffic signals could reduce accident frequency and severity.

Enhancing road safety during rush hours through measures like staggered work hours or improved public transportation could help mitigate accidents.

Focusing on urban areas for traffic safety campaigns and infrastructure improvements could address the higher accident rates in these regions.

Improving visibility conditions through better lighting and road design could reduce the severity of accidents.


summary and Conclution
summary :
1. Data does't contain the detail about whole states of teh Use , in real USA has 52 state whare as hear are only 45 are given
2. Less that 10 % of the cities are having higher accidenet than 1000 yearly
3. Most of the accidents are happends in the morning between 6am to 10 am and 3pm to 6pm in normal working days..
4. In Weekends most of the accidents are happends in between 10am to 3pm
5. much data is missing for the 2016 and thos missing data is basically due to the source 1 coz it's have very less data for 2016
6. Most of the Accidents are Happends in the Colder season or we can say it's happends in Winter ( when low temp .)
7. less than 35 % of the weather conditions are having higher accidents than 1000 yearly
Conclution :
This analysis describe that certain ciities , areas , times of day , day of week , temperatures , and seasons are having higher associated with higher accident frequencies

Many cities are having only one accidents so they need to be removal.. also thare is many row which have many empty values ( more than 40%) they should be consider to removal

Thare is some Data-Samppling issue with Dataset Also , source-1 colletcted less Data about year 2016 which makes this samplling issue in Data , it should to be consider to correct..

Further Exploration and Understanding of the Data could more help to understanding the factors which are contributing to accidents in different location and time..

Ask and Answer Questions :
Are thare more accident in warmer or colder area..
which 5 state have the highest accidents..
at which time accidents are most frequently..
which day of the week have most accidents...
which month have the most accidents..
What is the trend of the accident year by year ( Increasing / Decreasing ).. ?
What time of the day are accidents are most frequent

"""