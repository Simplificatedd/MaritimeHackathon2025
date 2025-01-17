import pandas as pd
from collections import Counter

# Path to the CSV file
file = "cleaned_data/train_cleaned.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file)

# Check if 'def_text' column exists
if 'def_text' not in df.columns:
    raise ValueError("The 'def_text' column does not exist in the CSV file.")

# Combine all text in the 'def_text' column into a single string
all_text = " ".join(df['def_text'].astype(str))

# Split the text into words and count their occurrences
word_counts = Counter(all_text.split())

# Convert the Counter object to a DataFrame
word_counts_df = pd.DataFrame(word_counts.items(), columns=["Word", "Count"])

# Sort the DataFrame by count in descending order
word_counts_df = word_counts_df.sort_values(by="Count", ascending=False)

# Save the results to a CSV file
output_file = "word_counts.csv"
word_counts_df.to_csv(output_file, index=False)

print(f"Word counts saved to {output_file}")