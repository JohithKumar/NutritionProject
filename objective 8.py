import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
plt.figure(figsize=(9,6))
sns.lineplot(data=df,x='Number_of_towns',y='Number_of_households',marker='o',color='blue')
plt.title('Number_of_towns vs Number_of_households')
plt.xlabel('Number_of_towns')
plt.ylabel('Number_of_households')
plt.show()
