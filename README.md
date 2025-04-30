# Task 1: Load and Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = sns.load_dataset('iris')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Data types and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# No missing values in this dataset, so no cleaning needed.

# Task 2: Basic Data Analysis

# Basic statistics
print("\nDescriptive statistics:")
print(df.describe())

# Group by species and calculate mean
grouped = df.groupby('species').mean()
print("\nAverage values grouped by species:")
print(grouped)

# Task 3: Data Visualization

# Set style for plots
sns.set(style='whitegrid')

# Line chart: simulate time series (use index as time for illustration)
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['sepal_length'], label='Sepal Length')
plt.title("Trend of Sepal Length (simulated over time)")
plt.xlabel("Index (Time)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal_length', data=df)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Sepal Width distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal_width'], bins=10, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# Scatter plot: Sepal Length vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Findings:
print("\nFindings:")
print("- Setosa species tends to have smaller petal length and width.")
print("- Sepal width has a somewhat normal distribution.")
print("- There is a visible positive correlation between sepal length and petal length.")


   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
4                5.0               3.6                1.4               0.2  setosa


setosa        1.462
versicolor    4.260
virginica     5.552
