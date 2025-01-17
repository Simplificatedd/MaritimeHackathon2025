import pandas as pd
from collections import Counter

# Path to the CSV file
file = "cleaned_data/train_cleaned.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file)

# Check if 'deficiency_code' column exists
if 'deficiency_code' not in df.columns:
    raise ValueError("The 'deficiency_code' column does not exist in the CSV file.")

# Combine all text in the 'deficiency_code' column into a single string
all_text = " ".join(df['deficiency_code'].astype(str))

# Split the text into words and count their occurrences
word_counts = Counter(all_text.split())

# Convert the Counter object to a DataFrame
word_counts_df = pd.DataFrame(word_counts.items(), columns=["Code", "Count"])

# Format the 'Code' column to ensure it is 5 digits with leading zeros
word_counts_df['Code'] = word_counts_df['Code'].apply(lambda x: x.zfill(5))

# Sort the DataFrame by count in descending order
word_counts_df = word_counts_df.sort_values(by="Count", ascending=False)

# Save the results to a CSV file
output_file = "misc/deficiency_code_counts.csv"
word_counts_df.to_csv(output_file, index=False)

print(f"Deficiency counts saved to {output_file}")