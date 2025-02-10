# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:28:31 2025

@author: harpr
"""
#%%
import numpy as np
import pandas as pd

#%%
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#%%
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df =pd.read_csv("diabetes_dataset.csv")

#%%
print("First 5 rows of the dataset:")
print(df.head())

#%%
print("\nDataset Info:")
print(df.info())

#%%
print("\nSummary Statistics:")
print(df.describe())

#%%
print("\nMissing Values:")
print(df.isnull().sum())

#%%
plt.figure(figsize=(6,4))
sns.countplot(x='Diabetes_Diagnosis', data=df, palette="coolwarm")
plt.title("Diabetes Diagnosis Distribution")
plt.xlabel("Diabetes Diagnosis (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

#%%
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True, color="red")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

#%%
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5,
            square=True)

plt.title("Correlation Heatmap of Numerical Variables")

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

for col in numeric_df.columns:
    print(f"- {col}")
    
#%%
plt.figure(figsize=(8,6))
sns.BoxPlotArtists(x="Diabetes_Diagnosis", y="BMI", data=df, palette="Set2")
plt.title("BMI vs Diabetes Diagnosis")
plt.xlabel("Diabetes Diagnosis")
plt.ylabel("BMI")
plt.show()

#%%
numeric_df = df.select_dtypes(include=['int64', 'float64'])

correlation = numeric_df.corr()["Diabetes_Diagnosis"].sort_values(ascending=False)
print(correlation.head(10))

plt.figure(figsize=(10, 6))
top_features = correlation[1:11]  
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Features Correlated with Diabetes Diagnosis")
plt.xlabel("Correlation Coefficient")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(correlation.apply(lambda x: f"{x:.3f} ({'Strong' if abs(x) > 0.5 else 'Moderate' if abs(x) > 0.3 else 'Weak'})")
      .head(10))
#%%

#%%

#%%

#%%

#%%    
