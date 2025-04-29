# src/load_data.py
import pandas as pd

# Load the dataset and drop the unnecessary column
df = pd.read_csv('../data/labeled_data.csv')
df = df.drop(columns=['Unnamed: 0'])  # Drop the unwanted index column

# Show basic info again
print("First few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nClass distribution:")
print(df['class'].value_counts())


