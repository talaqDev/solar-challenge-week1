import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_quality_check(data, dataset_name):
    print(f"Data Quality Check for {dataset_name}:")
    print(data.info())  
    print("Missing values:\n", data.isnull().sum())
    print("Negative GHI values:\n", data[data['GHI'] < 0]) 
    print("\n")



def summary_statistics(data, dataset_name):
    print(f"Summary Statistics for {dataset_name}:")
    print(data.describe())  
    print("\n")


def main():
    benin_data = pd.read_csv('../data/benin-malanville.csv')  
    sierra_leone_data = pd.read_csv('../data/sierraleone-bumbuna.csv')
    togo_data = pd.read_csv('../data/togo-dapaong_qc.csv')

    datasets = {
        "Benin": benin_data,
        "Sierra Leone": sierra_leone_data,
        "Togo": togo_data,
    }

    for name, data in datasets.items():
        data_quality_check(data, name)
        summary_statistics(data, name)
main()
