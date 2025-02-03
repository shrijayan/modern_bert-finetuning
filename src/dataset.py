import json
import csv
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract values from config
train_file = config['train_file']
test_file = config['test_file']
valid_file = config['valid_file']
model_id = config['base_model']
task_type = config['task_type']
dataset_id = config['dataset_id']
MAX_TOKEN_LENGTH = config['MAX_TOKEN_LENGTH']
dataset_csv = config['dataset_csv']

def load_raw_dataset():
    dataset = load_dataset("csv", data_files=dataset_csv)
    return dataset

def get_label_names(dataset_csv):
    unique_labels = set()
    with open(dataset_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'label' in row:
                unique_labels.add(row['label'])
    return sorted(list(unique_labels))

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = MAX_TOKEN_LENGTH
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, return_tensors="pt")
    return dataset.map(tokenize, batched=True, remove_columns=["text"])

def load_and_preprocess_dataset(model_id, task_type):
    raw_dataset = load_raw_dataset()
    tokenizer = get_tokenizer(model_id)
    label_names = get_label_names(dataset_csv)
    label2id = {label: i for i, label in enumerate(label_names)}

    if task_type in ["binary", "multiclass"]:
        raw_dataset = raw_dataset.map(lambda example: {'label': label2id.get(example['label'], -1)})
    elif task_type == "multilabel":
        raw_dataset = raw_dataset.map(lambda example: {'label': [label2id.get(label, -1) for label in example['label'].split(',')]})

    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    return tokenized_dataset, label_names
    tokenizer = get_tokenizer(model_id)
    label_names = get_label_names(dataset_csv)
    label2id = {label: i for i, label in enumerate(label_names)}
    raw_dataset = raw_dataset.map(lambda example: {'label': label2id[example['label']]})
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    return tokenized_dataset, label_names

if __name__ == "__main__":
    tokenized_dataset = load_and_preprocess_dataset(model_id, task_type)
    print(tokenized_dataset["train"].features.keys())