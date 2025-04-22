import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
#OBJECTIVE 2 (BAR CHART) Population Distribution Across States using Bar Chart
states = df[(df["Region"] == "STATE") & (df["Area_type"] == "Total")]
states.loc[:, "Population_Persons"] = pd.to_numeric(states["Population_Persons"], errors="coerce")
states = states.sort_values("Population_Persons", ascending=True)
plt.figure(figsize=(9, 6))
plt.barh(states["Name"], states["Population_Persons"], color='skyblue')
plt.xlabel("Population")
plt.title("Population Distribution Across States")
plt.show()
