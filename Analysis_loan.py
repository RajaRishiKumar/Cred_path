import pandas as pd

# Read the CSV file
data = pd.read_csv('loan.csv')

# Show first few rows
print("Data Preview:\n", data.head())

# Show shape of the dataset
print("\nData Shape:", data.shape)

# Example: Average of a column (if your CSV has a column named 'Score')
if 'Score' in data.columns:
    print("\nAverage Score:", data['Score'].mean())
