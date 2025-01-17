import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define the path to your locally fine-tuned model
model_path = "pytorch-fine-tuned-model"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained(model_path)

# Set the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DataFrame
df = pd.read_csv('input.csv')

# Duplicate the DataFrame
df_manipulate = df.copy()

# Remove specified columns
columns_to_remove = ["PscInspectionId", "annotation_id", "username", "InspectionDate", "VesselId", "PscAuthorityId"]
if set(columns_to_remove).issubset(df.columns):
    df_manipulate = df.drop(columns=columns_to_remove)

# Define the split_info function
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
if 'def_text' in df_manipulate.columns:
    # Split the 'def_text' column into new columns
    new_columns = df_manipulate.apply(split_info, axis=1)
    # Add the new columns to the DataFrame
    df_manipulate = pd.concat([df_manipulate, new_columns], axis=1)
    # Drop the original 'def_text' column after splitting
    df_manipulate = df_manipulate.drop(columns=['def_text'])

# Remove columns where all values are NaN (empty cells)
df_manipulate = df_manipulate.dropna(axis=1, how='all')

# Drop unnecessary columns
df_manipulate = df_manipulate.drop(columns=["PscInspectionId", "Deficiency Code"])

# Define the predict function
def predict(texts, model, tokenizer, device, max_len=512):
    model.eval()
    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1).tolist()
    return predicted_classes

# Example usage of predict function
def generate_prompt(row):
    # Convert each row into a string prompt
    prompt = ""
    for col, value in row.items():
        prompt += f"{col}: {value}\n"
    return prompt.strip()

# Generate prompts for each row
prompts = df_manipulate.apply(generate_prompt, axis=1).tolist()

# Predict severity for each prompt
predictions = predict(prompts, model, tokenizer, device)

# Add predictions to the original DataFrame
df['severity'] = predictions

# Save the updated DataFrame to a CSV file
df.to_csv('output.csv', index=False)