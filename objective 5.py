import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
#OBJECTIVE 4(PAIR PLOT) Sub-District-wise Relationship Analysis using Pair Plot
df.columns = df.columns.str.strip()
plt.figure(figsize=(9,6))
data = df[df["Region"] == "SUB-DISTRICT"][[
    "Population_Persons", 
    "Number_of_households", 
    "Area", 
    "Population_per_sq_km."]]
data = data.apply(pd.to_numeric, errors='coerce').dropna()
sns.pairplot(data)
plt.suptitle("Sub-Districts: Population, Households, Area & Density", y=1.02)
plt.show()
