from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Step 1: Load your CSV files
train_csv_path = "cleaned_data/training.csv"  # Replace with your training CSV file path
val_csv_path = "cleaned_data/validation.csv"  # Replace with your validation CSV file path

# Load CSV files into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Step 2: Convert DataFrames to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Verify the dataset columns
print("Training Dataset Columns:", train_dataset.column_names)
print("Validation Dataset Columns:", val_dataset.column_names)

# Step 3: Load the pre-trained model and tokenizer
model_name = "Qwen/Qwen2.5-3B"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Determine the number of unique classes in annotation_severity
num_labels = train_df["annotation_severity"].nunique()
print(f"Number of unique classes: {num_labels}")

# Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


# Step 4: Tokenize the dataset
def preprocess_function(examples):
    # Combine all columns except 'annotation_severity' into a single input string
    input_columns = [col for col in examples.keys() if col != "annotation_severity"]
    inputs = [" ".join([str(examples[col][i]) for col in input_columns]) for i in range(len(examples[input_columns[0]]))]
    
    # Tokenize the inputs
    tokenized_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=512)
    
    # Add labels (annotation_severity)
    tokenized_inputs["labels"] = examples["annotation_severity"]
    
    return tokenized_inputs

# Apply tokenization to the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

print("Tokenized Train Dataset Example:", tokenized_train_dataset[0])

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",  # Directory to save the fine-tuned model
    eval_strategy="epoch",           # Evaluate at the end of each epoch (replaces evaluation_strategy)
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=1,   # Batch size for training
    per_device_eval_batch_size=1,    # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Weight decay for regularization
    save_strategy="epoch",           # Save model at the end of each epoch
    logging_dir="./logs",            # Directory for logs
    logging_steps=10,                # Log every 10 steps
    fp16=False,                      # Disable fp16 for MPS devices
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
)

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model and tokenizer
trainer.save_model("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

print("Model training complete and saved to './fine-tuned-model'")