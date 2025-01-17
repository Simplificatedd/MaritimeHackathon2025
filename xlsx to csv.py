import pandas as pd

# Path to the Excel file
input_file = "cleaned_data/deficiency_codes.xlsx"

# Path to the output CSV file
output_file = "cleaned_data/deficiency_codes.csv"

# Read the Excel file into a DataFrame
df = pd.read_excel(input_file)

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)