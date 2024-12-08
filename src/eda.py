import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes


def data_quality_check(data, dataset_name):
    print(f"Data Quality Check for {dataset_name}:")
    print("Missing values:\n", data.isnull().sum())
    print("Negative GHI values:\n", data[data['GHI'] < 0]) 
    print("\n")



def summary_statistics(data, dataset_name):
    print(f"Summary Statistics for {dataset_name}:")
    print(data.describe())  
    print("\n")
def radiation_over_time(data, dataset_name):
    print(f"Plot GHI, DNI, Tamb and DHI over time for {dataset_name}:")
    plt.figure(figsize=(14, 7))
    plt.plot(data['GHI'], label='GHI', alpha=0.8)
    plt.plot(data['DNI'], label='DNI', alpha=0.8)
    plt.plot(data['DHI'], label='DHI', alpha=0.8)
    plt.plot(data['Tamb'], label='Tamb', alpha=0.8)
    plt.title(f'Solar Radiation Over Time for {dataset_name}:')
    plt.xlabel('Time')
    plt.ylabel('Radiation (W/m²)')
    plt.legend()
    plt.show()
def monthly_average_radiation(data, data_name):
    print(f'Monthly average solar radiation of {data_name}:')
    plt.figure(figsize=(14, 7))
    plt.plot(data['GHI'], label='Monthly Average GHI', marker='o')
    plt.plot(data['DNI'], label='Monthly Average DNI', marker='o')
    plt.plot(data['DHI'], label='Monthly Average DHI', marker='o')
    plt.title(f'Monthly Average Solar Radiation of {data_name}')
    plt.xlabel('Month')
    plt.ylabel('Radiation (W/m²)')
    plt.legend()
    plt.show()


def plot_correlation_matrix(data, dataset_name, variables):
  
    # Calculate correlation matrix
    corr = data[variables].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix - {dataset_name}')
    plt.show()

def plot_pairplot(data, dataset_name, variables):
  
    sns.pairplot(data[variables], diag_kind='kde', corner=True)
    plt.suptitle(f'Pair Plot - {dataset_name}', y=1.02)
    plt.show()
def plot_wind_vs_radiation(data, dataset_name):
 
    plt.figure(figsize=(14, 7))
    plt.scatter(data['WS'], data['GHI'], alpha=0.7, label='WS vs GHI', color='blue')
    plt.scatter(data['WSgust'], data['GHI'], alpha=0.7, label='WSgust vs GHI', color='green')
    plt.title(f'Wind Speed vs Solar Irradiance - {dataset_name}')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('GHI (W/m²)')
    plt.legend()
    plt.show()


def plot_wind_rose(data, dataset_name):
   
    plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax()
    ax.bar(data['WD'], data['WS'], normed=True, opening=0.8, edgecolor='white')
    ax.set_title(f'Wind Rose - {dataset_name}')
    plt.show()

def plot_wind_direction_variability(data, dataset_name):

    
    bins = np.arange(0, 361, 45)
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    data['WindDirectionBin'] = pd.cut(data['WD'], bins=bins, labels=labels, right=False)

    
    direction_counts = data['WindDirectionBin'].value_counts().sort_index()

    
    angles = np.linspace(0, 2 * np.pi, len(direction_counts), endpoint=False).tolist()
    angles += angles[:1] 

    values = direction_counts.tolist()
    values += values[:1] 

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.bar(angles, values, width=0.4, color='blue', edgecolor='black', alpha=0.7)
    ax.set_theta_offset(np.pi / 2) 
    ax.set_theta_direction(-1)  
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f'Wind Direction Variability - {dataset_name}')
    plt.show()
def plot_temperature_humidity_correlation(data, dataset_name, variables):
    corr = data[variables].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Between RH, Temperature, and Radiation - {dataset_name}')
    plt.show()

def plot_scatter_rh_temperature(data, dataset_name):

    plt.figure(figsize=(14, 7))
    
    # RH vs Temperature (Tamb)
    plt.subplot(1, 2, 1)
    plt.scatter(data['RH'], data['Tamb'], alpha=0.6, color='blue')
    plt.title(f'RH vs Tamb - {dataset_name}')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Ambient Temperature (°C)')
    
    # RH vs GHI
    plt.subplot(1, 2, 2)
    plt.scatter(data['RH'], data['GHI'], alpha=0.6, color='orange')
    plt.title(f'RH vs GHI - {dataset_name}')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    
    plt.tight_layout()
    plt.show()

def plot_histograms(data, dataset_name, variables, bins=20):
 
    for var in variables:
        if var in data.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(data[var].dropna(), bins=bins, color='blue', alpha=0.7, edgecolor='black')
            plt.title(f'{var} Distribution - {dataset_name}')
            plt.xlabel(var)
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()



def main():
    benin_data = pd.read_csv('../data/benin-malanville.csv')  
    sierra_leone_data = pd.read_csv('../data/sierraleone-bumbuna.csv')
    togo_data = pd.read_csv('../data/togo-dapaong_qc.csv')

    datasets = {
        "Benin": benin_data,
        "Sierra Leone": sierra_leone_data,
        "Togo": togo_data
    }
    compare = ['GHI', 'DNI', 'DHI', 'Tamb', 'ModA', 'ModB', 'TModA', 'TmodB']
    radiation_temp_vars = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    benin_data['WS'] = benin_data['WS'].apply(lambda x: np.nan if x < 0 else x)
    benin_data['WD'] = benin_data['WD'].apply(lambda x: np.nan if x < 0 else x)
    temp_humidity_vars = ['RH', 'Tamb', 'TModA', 'TModB', 'GHI', 'DNI', 'DHI']
    variables_to_plot = ['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'Tamb', 'TModA', 'TModB']

        
    #Dataset
    benin_data['Timestamp'] = pd.to_datetime(benin_data['Timestamp'])
    benin_data.set_index('Timestamp', inplace=True)
    sierra_leone_data['Timestamp'] = pd.to_datetime(sierra_leone_data['Timestamp'])
    sierra_leone_data.set_index('Timestamp', inplace=True)
    togo_data['Timestamp'] = pd.to_datetime(togo_data['Timestamp'])
    togo_data.set_index('Timestamp', inplace=True)

  
    sierra_leone_monthly = sierra_leone_data.resample('ME').mean()
    togo_monthly = togo_data.resample('ME').mean()
    benin_monthly = benin_data.resample('ME').mean()
    def comparision(variable):
        print(f'Comparison of {variable}')
        plt.figure(figsize=(14, 7))
        plt.plot(benin_monthly[variable], label=f'Benin {variable}', marker='o')
        plt.plot(sierra_leone_monthly[variable], label=f'Sierra Leone {variable}', marker='o')
        plt.plot(togo_monthly[variable], label=f'Togo {variable}', marker='o')
        plt.title(f'Comparison of Monthly Average {variable} Across Locations')
        plt.xlabel('Month')
        plt.ylabel(f'{variable} (W/m²)')
        plt.legend()
        plt.show()

    for name, data in datasets.items():
        data_quality_check(data, name)
        summary_statistics(data, name)
        radiation_over_time(data, name)
        monthly_average_radiation(data, name)
        plot_correlation_matrix(data, name, radiation_temp_vars)
        plot_pairplot(data, name, radiation_temp_vars)
        plot_wind_vs_radiation(data, name)
        plot_wind_rose(data, name)
        plot_wind_direction_variability(data, name)
        plot_temperature_humidity_correlation(data, name, temp_humidity_vars)
        plot_scatter_rh_temperature(data, name)
        plot_histograms(data, name, variables_to_plot)
    for var in compare:
        comparision(var)
        pass
main()