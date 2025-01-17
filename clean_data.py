import pandas as pd
import numpy as np

# Load the CSV files into DataFrames
test_df = pd.read_csv('dataset/psc_severity_test.csv')
train_df = pd.read_csv('dataset/psc_severity_train.csv')

# Drop rows with any empty cells
test_df = test_df.dropna()
train_df = train_df.dropna()

# Sort the DataFrames by 'PscInspectionId' in ascending order
test_df = test_df.sort_values(by='PscInspectionId', ascending=True)
train_df = train_df.sort_values(by='PscInspectionId', ascending=True)

# Save the cleaned and sorted DataFrames to CSV files
test_df.to_csv('cleaned_data/test_cleaned.csv', index=False)
train_df.to_csv('cleaned_data/train_cleaned.csv', index=False)