import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
train_df = pd.read_csv('cleaned_data/train_cleaned.csv')

# List of columns to remove
columns_to_remove = ["PscInspectionId", "annotation_id", "username", "InspectionDate", "VesselId", "PscAuthorityId"]

# Drop the specified columns
if set(columns_to_remove).issubset(train_df.columns):
    train_df = train_df.drop(columns=columns_to_remove)

# Function to split the information into columns
def split_info(row):
    info_dict = {}
    lines = row['def_text'].split('\n')
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            info_dict[key] = value
        elif line.startswith('*'):
            if 'Deficiency/Finding' in info_dict:
                info_dict['Deficiency/Finding'] += '\n' + line
            else:
                info_dict['Deficiency/Finding'] = line
    return pd.Series(info_dict)

# Apply the function to split the data (assuming the column with the data is named 'def_text')
if 'def_text' in train_df.columns:
    # Split the 'def_text' column into 9 new columns
    new_columns = train_df.apply(split_info, axis=1)
    # Add the new columns to the DataFrame
    train_df = pd.concat([train_df, new_columns], axis=1)
    # Drop the original 'def_text' column after splitting
    train_df = train_df.drop(columns=['def_text'])

# Remove columns where all values are NaN (empty cells)
train_df = train_df.dropna(axis=1, how='any')

train_df = train_df.drop(columns=["PscInspectionId", "Deficiency Code"])

# Save the cleaned and processed DataFrame to a new CSV file
train_df.to_csv('cleaned_data/training.csv', index=False)

print("Columns removed, data split into new columns, empty columns removed, 'def_text' column removed, and file saved successfully.")