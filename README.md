# BigData-Project
---

# 🚗 US Accidents Analysis with PySpark

This project uses **PySpark** to perform large-scale data processing and extract insightful trends from the **US Accidents Dataset** sourced from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data). It demonstrates the power of distributed data processing to handle and analyze millions of records efficiently.

---


#### 📊 Dataset (Updated)

- **Name**: US Accidents (7.7 million records)
- **Source**: [Kaggle - US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data)
- **Size**: ~**3.06 GB** CSV file
- **Scope**: Covers ~7.7 million accidents across the US from 2016 to 2023
- **Features**: 
  - 47 columns including `Severity`, `Start_Time`, `City`, `State`, `Temperature`, `Visibility`, `Weather_Condition`, etc.
  - Many fields have **thousands of unique values** (e.g., over 350 unique weather conditions, thousands of city/street names)
  
#### ⚠️ Note

Due to its size and number of unique entries, this dataset is a **great fit for PySpark**, as it:
- Exceeds the comfortable range for in-memory Pandas processing
- Benefits from parallelized data transformations and filtering
- Can reveal insights at scale (e.g., trends over time, geographic clustering, severity prediction)

---

## 🔧 Project Structure

```bash
us-accidents-pyspark/
│
├── data/
│   └── US_Accidents.csv          # Downloaded dataset
│
├── notebooks/
│   └── exploration.ipynb         # Initial EDA using PySpark
│
├── src/
│   ├── preprocessing.py          # Data cleaning & transformation
│   ├── insights.py               # Spark jobs for analysis
│   └── utils.py                  # Helper functions
│
├── output/
│   └── visualizations/           # Plots and charts
│
├── requirements.txt              # Dependencies
├── README.md                     # Project overview
└── run_analysis.py               # Main script to run analysis
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Person-Ad/BigData-Project
cd BigData-Project
```

### 2. Install dependencies

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have **PySpark** installed:

```bash
pip install pyspark
```

### 3. Download the dataset

Download `US_Accidents.csv` from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data) and place it inside the `data/` directory.

### 4. Run the analysis

```bash
python run_analysis.py
```

---

## 📈 Key Analyses

- Accident distribution by **state**, **city**, **time**, and **weather**
- **Top 10 accident-prone cities**
- Accidents by **severity level**
- Impact of **weather conditions** (rain, snow, fog, etc.)
- Temporal trends (hour of day, day of week, monthly)
- Heatmaps and visualizations

---

## 🛠 Technologies Used

- **PySpark** – distributed data processing
- **Pandas/Matplotlib/Seaborn** – for comparative visualization
- **Jupyter Notebook** – for exploratory data analysis
- **Python 3.8+**

---

## 📌 Sample Insights

- California and Florida report the most accidents.
- Majority of accidents occur during rush hours (7-9 AM, 4-6 PM).
- Rain and foggy weather increase the likelihood of severe accidents.
- Most accidents are of **Severity 2** (on a 1–4 scale).

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)

---

## 🤝 Contributions

Contributions are welcome! Open an issue or submit a PR to enhance this project.

---

## 📄 License

This project is licensed under the MIT License.

---

