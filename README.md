**Solar Energy Analysis Project**

**Project Overview**

This project performs comprehensive data analysis and visualization on solar energy datasets from various regions (e.g., Benin, Sierra Leone, Togo). The goal is to understand solar radiation patterns, wind conditions, temperature influences, and other environmental factors using time series analysis, correlation analysis, and machine learning techniques. The project also includes data cleaning and outlier detection for accurate insights.

**Key Features**

Data Quality Check: Identifies missing values, outliers, and handles anomalies.

Exploratory Data Analysis (EDA): Includes visualizations like time series, histograms, and correlation matrices.

Wind Analysis: Uses wind roses and scatter plots to visualize wind speed and direction.

Temperature Analysis: Analyzes the influence of relative humidity on temperature and solar radiation.

Bubble Charts: Visualizes relationships between solar radiation, temperature, wind speed, and humidity.

Z-Score Analysis: Identifies significant outliers based on Z-scores.

Data Cleaning: Handles missing data, drops unnecessary columns, and manages anomalies.

**Technologies Used**

Python: Programming language

pandas: Data manipulation and analysis

numpy: Numerical operations

matplotlib: Data visualization

seaborn: Statistical data visualization

windrose: Wind rose visualizations

scipy: Statistical functions (e.g., Z-score calculation)

**Installation**
1. Clone the repository

Clone the project repository to your local machine:

bash

Copy code

  git clone https://github.com/Hunegn/Moonlit.git

2. Install dependencies
Make sure you have a Python 3 environment set up. Install the required libraries using requirements.txt:

bash

Copy code

pip install -r requirements.txt

This will install all the necessary libraries, including:

pandas
numpy
matplotlib
seaborn
windrose
scipy

**3. Virtual Environment (Optional but Recommended)**
If you’re using a virtual environment, create and activate it before installing dependencies:

bash

Copy code

# For Linux/macOS
python3 -m venv env
source env/bin/activate

# For Windows
python -m venv env
.\env\Scripts\activate
Usage
**Running the Analysis**
After installing the necessary dependencies, you can run the analysis using the main() function in the main.py file:

Ensure you have the necessary dataset files (benin-malanville.csv, sierraleone-bumbuna.csv, togo-dapaong_qc.csv) in the data/ folder.
Run the analysis:
bash
Copy code
python main.py
This will:

**Perform data quality checks.**

Generate visualizations such as time series plots, histograms, and correlation matrices.

Perform wind analysis and temperature/humidity correlation.

Clean and save the cleaned datasets.

Saving Cleaned Data

The cleaned datasets will be saved as CSV files (e.g., benin_cleaned.csv).

**Customizing the Analysis**

You can customize the analysis by modifying:

The dataset file paths.

The variables used for correlation and visualization.

The parameters for the plots (e.g., bin size for histograms).


       
