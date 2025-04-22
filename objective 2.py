import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df=pd.read_csv(r'C:/Users/KOLLA/OneDrive/Desktop/PYTHONDATASET.csv')
#OBJECTIVE 1  (SCATTER PLOT)
df["Population_Males"] = pd.to_numeric(df["Population_Males"], errors="coerce")
df["Population_Females"] = pd.to_numeric(df["Population_Females"], errors="coerce")
df_clean = df.dropna(subset=["Population_Males", "Population_Females"])
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_clean,
    x="Population_Males",
    y="Population_Females",
    color="blue",
)
plt.title("Scatter Plot of Male vs Female Population")
plt.xlabel("Population Males")
plt.ylabel("Population Females")
plt.show()
