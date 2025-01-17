import pandas as pd

file_path = 'train_cleaned.csv'
data = pd.read_csv(file_path)

print(data.head())
