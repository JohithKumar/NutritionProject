import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
#OBJECTIVE 3(PIE CHART) Urban vs Rural Population Distribution in India using pie chart
data = df[(df["Region"] == "INDIA") & (df["Area_type"].isin(["Rural", "Urban"]))]
colors = ["lightgreen", "deepskyblue"]
plt.figure(figsize=(9,6))
plt.pie(data["Population_Persons"], labels=data["Area_type"],  autopct="%1.1f%%",colors=colors, startangle=90)
plt.title("Urban vs Rural Population in India")
plt.axis("equal")
plt.show()
