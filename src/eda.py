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
    
    #Dataset
    benin_data['Timestamp'] = pd.to_datetime(benin_data['Timestamp'])
    benin_data.set_index('Timestamp', inplace=True)
    sierra_leone_data['Timestamp'] = pd.to_datetime(sierra_leone_data['Timestamp'])
    sierra_leone_data.set_index('Timestamp', inplace=True)
    togo_data['Timestamp'] = pd.to_datetime(togo_data['Timestamp'])
    togo_data.set_index('Timestamp', inplace=True)

    # Monthly averages for Sierra Leone and Togo
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
        pass
    for var in compare:
        comparision(var)
main()