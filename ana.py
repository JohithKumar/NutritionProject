import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Nutri.csv')

print("Preview of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nNumber of Unique Values in Each Column:")
print(df.nunique())

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Full Dataset)')
plt.tight_layout()
plt.show()

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    df[col].hist(bins=20)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

cat_cols = df.select_dtypes(include='object').columns
if not cat_cols.empty and not numeric_columns.empty:
    sns.boxplot(x=cat_cols[0], y=numeric_columns[0], data=df)
    plt.title(f'Boxplot of {numeric_columns[0]} by {cat_cols[0]}')
    plt.tight_layout()
    plt.show()

if len(numeric_columns) <= 5:
    sns.pairplot(df[numeric_columns])
    plt.show()

df_clean = df.dropna()

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

if not cat_cols.empty and not numeric_columns.empty:
    grouped = df.groupby(cat_cols[0])[numeric_columns[0]].mean()
    print(f"\nAverage {numeric_columns[0]} grouped by {cat_cols[0]}:")
    print(grouped)
