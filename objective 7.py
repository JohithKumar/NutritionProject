import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
df["Number_of_households"] = pd.to_numeric(df["Number_of_households"], errors="coerce")
plt.figure(figsize=(9, 6))
sns.boxplot(x="Region", y="Number_of_households", data=df, color="salmon")  
plt.title("Distribution of Number of Households by Region")
plt.xlabel("Region")
plt.ylabel("Number of Households")
plt.show()
