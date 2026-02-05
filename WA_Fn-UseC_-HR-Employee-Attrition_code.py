import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel('WA_Fn-UseC_-HR-Employee-Attrition.xlsx')

# Quick inspection
print(df.head())
print(df.info())  # Checks for null values and data types

# Drop useless columns
df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Check for duplicates
print(df.duplicated().sum())

sns.countplot(x='Attrition', data=df)
plt.title('Attrition Distribution')
plt.show()

# Set a nice aesthetic for all plots
sns.set_theme(style="whitegrid")

# --- Visualization 1: Income vs. Attrition (Boxplot) ---
# Purpose: See if people who leave (Yes) generally have a lower median income than those who stay (No).
plt.figure(figsize=(8, 6))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='coolwarm')
plt.title('Monthly Income vs. Attrition Status')
plt.xlabel('Left Company (Attrition)')
plt.ylabel('Monthly Income')
plt.show()

# --- Visualization 2: Overtime vs. Attrition (Countplot) ---
# Purpose: Compare the count of people leaving who work overtime vs. those who don't.
plt.figure(figsize=(8, 6))
sns.countplot(x='OverTime', hue='Attrition', data=df, palette='viridis')
plt.title('Attrition Rates by OverTime Status')
plt.xlabel('Works OverTime?')
plt.ylabel('Count of Employees')
plt.legend(title='Attrition')
plt.show()

# --- Visualization 3: Age Distribution (Histogram with KDE) ---
# Purpose: Check if younger employees are leaving at a higher frequency than older ones.
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Attrition', kde=True, element="step", palette='magma')
plt.title('Age Distribution by Attrition Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# --- Visualization 4: Job Role Turnover (Countplot) ---
# Purpose: Identify which specific job titles have the highest "Yes" counts.
plt.figure(figsize=(14, 6))  # Wider figure to fit job titles
sns.countplot(x='JobRole', hue='Attrition', data=df, palette='Set2')
plt.title('Attrition by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Count of Employees')
plt.xticks(rotation=45)  # Rotates labels so they don't overlap
plt.legend(title='Attrition', loc='upper right')
plt.tight_layout() # Adjusts layout so labels aren't cut off
plt.show()

from sklearn.preprocessing import LabelEncoder

# Label Encoding for binary/ordinal columns
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

# One-Hot Encoding for other categoricals
df = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(15, 10))
# Calculate correlation of all columns with 'Attrition' and sort them
correlation = df.corr()['Attrition'].sort_values(ascending=False)

# Plot the top positive and negative correlations
correlation.drop('Attrition').plot(kind='bar')
plt.title('Feature Correlation with Attrition')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Split Data
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
