import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
plt.figure(figsize=(9,6))
corr = df[['Number_of_towns', 'Population_per_sq_km.','Number_of_households','Population_Persons']].corr()
print(corr)
colors=["lightgreen","orange"]
sns.heatmap(corr, annot=True, cmap=colors, fmt='.2f', linewidths=5)  
plt.title("Heatmap for Number_of_towns vs Population_per_sq_km.")
plt.show()
