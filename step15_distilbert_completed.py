import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
# Load Dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Use only first 5000 reviews
df = df.head(5000)

print(df.head())

print("\nDataset Shape:")
print(df.shape)
# Convert labels to numbers

df['label'] = df['sentiment'].map({
    'negative': 0,
    'positive': 1
})

print(df[['sentiment', 'label']].head())
# Train-Test Split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'],
    df['label'],
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(train_texts))
print("Testing Samples:", len(test_texts))
# Load DistilBERT Tokenizer

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# Tokenization Function

def tokenize_function(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128
    )
train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

print("Tokenization Completed")
# Create Dataset Objects

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": list(train_labels)
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": list(test_labels)
})

print("Datasets Created Successfully")
# Load DistilBERT Model

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

print("Model Loaded Successfully")
training_args = TrainingArguments(
    output_dir="./results",

    num_train_epochs=1,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    logging_steps=100,

    save_strategy="no"
)

print("Training Arguments Ready")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("Trainer Created Successfully")
print("Starting DistilBERT Training...")

trainer.train()

print("Training Completed Successfully")
print("Evaluating Model...")

results = trainer.evaluate()

print(results)