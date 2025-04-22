import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
print("DATASET: ",df)
print("DATASET without na values",df.dropna())
print(df.head())
print(df.describe())
print(df.isnull().sum())
