import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

# Load CSV files
train_csv_path = "cleaned_data/training.csv"  # Replace with your training CSV file path
val_csv_path = "cleaned_data/validation.csv"  # Replace with your validation CSV file path

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Combine datasets for tokenization
full_data = pd.concat([train_df, val_df], axis=0)

# Prepare features (X) and labels (y)
X = full_data.drop("annotation_severity", axis=1).apply(lambda row: " ".join(map(str, row)), axis=1).tolist()
y = full_data["annotation_severity"].tolist()

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load tokenizer
model_name = "PowerInfer/SmallThinker-3B-Preview"  # Replace with the desired model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_data(texts, labels, tokenizer, max_len=512):
    tokenized_data = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long)
}

# Tokenize train and validation data
train_data = tokenize_data(X_train, y_train, tokenizer)
val_data = tokenize_data(X_val, y_val, tokenizer)

class TextDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Create datasets
train_dataset = TextDataset(train_data)
val_dataset = TextDataset(val_data)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_labels = len(set(y)) # number of unique labels
print(f"Unique labels: {set(y)}, Number of labels: {num_labels}")

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)
num_labels = len(label_encoder.classes_)
print(f"Label mapping: {dict(enumerate(label_encoder.classes_))}")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

from torch.optim import AdamW
from transformers import get_scheduler

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Scheduler
num_training_steps = len(train_loader) * 2  # 2 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, epochs=3):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training Phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        print(f"Training Loss: {train_loss / len(train_loader)}")
        
        # Validation Phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        print(f"Validation Loss: {val_loss / len(val_loader)}")
        print(f"Validation Accuracy: {correct / total}")


train_model(model, train_loader, val_loader, optimizer, lr_scheduler, loss_fn, epochs=3)

model.save_pretrained("./smallthinker-fine-tuned-model")
tokenizer.save_pretrained("./smallthinker-fine-tuned-model")

print("Model and tokenizer saved.")
