import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 


def data_quality_check(data, dataset_name):
    print(f"Data Quality Check for {dataset_name}:")
    print("Missing values:\n", data.isnull().sum())
    print("Negative GHI values:\n", data[data['GHI'] < 0]) 
    print("\n")



def summary_statistics(data, dataset_name):
    print(f"Summary Statistics for {dataset_name}:")
    print(data.describe())  
    print("\n")
def time_series_analysis(data, dataset_name):
    print(f"Plot GHI, DNI, Tamb and DHI over time for {dataset_name}:")
    plt.figure(figsize=(14, 7))
    plt.plot(data['GHI'], label='GHI', alpha=0.8)
    plt.plot(data['DNI'], label='DNI', alpha=0.8)
    plt.plot(data['DHI'], label='DHI', alpha=0.8)
    plt.plot(data['Tamb'], label='Tamb', alpha=0.8)
    plt.title(f'Solar Radiation Over Time {dataset_name}:')
    plt.xlabel('Time')
    plt.ylabel('Radiation (W/m²)')
    plt.legend()
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
    benin_data['Timestamp'] = pd.to_datetime(benin_data['Timestamp'])
    benin_data.set_index('Timestamp', inplace=True)
    for name, data in datasets.items():
        pass
        #data_quality_check(data, name)
        #summary_statistics(data, name)
        time_series_analysis(data, name)
   



main()