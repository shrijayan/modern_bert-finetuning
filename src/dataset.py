import json
import csv
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import ast

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract values from config
model_id = config['base_model']
task_type = config['task_type']
dataset_id = config['dataset_id']
MAX_TOKEN_LENGTH = config['MAX_TOKEN_LENGTH']
dataset_csv = config['dataset_csv']

def load_raw_dataset():
    dataset = load_dataset("csv", data_files=dataset_csv, index_col=False)
    return dataset

def get_label_names(dataset_csv):
    unique_values = set()
    with open(dataset_csv, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        try:
            headers = next(reader)
        except StopIteration:
            return []
        try:
            col_index = headers.index('labels')
        except ValueError:
            return []
        for row in reader:
            if col_index < len(row):
                label_str = row[col_index]
                try:
                    # Safely evaluate the string as a Python list
                    label_list = ast.literal_eval(label_str)
                    if isinstance(label_list, list): # Ensure it's actually a list
                        for label in label_list:
                            unique_values.add(label)
                except (ValueError, SyntaxError):
                    # Handle cases where the string is not a valid list representation
                    print(f"Warning: Could not parse labels string: {label_str} in row: {row}")
                    continue # Or handle it as needed (e.g., skip the row, log error)
    return list(unique_values)

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, return_tensors="pt")
    return dataset.map(tokenize, batched=True, remove_columns=["text"])

def split_dataset(dataset):
    train_testvalid = dataset['train'].train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    return DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']
    })

def load_and_preprocess_dataset(model_id, task_type):
    raw_dataset = load_raw_dataset()
    # for row in raw_dataset['train']:
    #     print(f"Text: {row['text']}, Label: {row['labels']}")
    tokenizer = get_tokenizer(model_id)
    # print(f"tokenizer: {tokenizer}")
    label_names = get_label_names(dataset_csv)
    # print(f"label_names: {label_names}")
    
    label2id = {label: i for i, label in enumerate(label_names)}
    # print(f"label2id: {label2id}")

    if task_type in ["binary", "multiclass"]:
        raw_dataset = raw_dataset.map(lambda example: {'labels': float(label2id.get(example['labels'], -1))})
        # print(f"raw_dataset: {raw_dataset["train"]["labels"]}")
        
    elif task_type == "multilabel":
        def map_labels_multilabel(example):
            label_str = example['labels'] # Get the string representation of the label list
            try:
                label_list = ast.literal_eval(label_str) # Parse the string into a Python list
            except (ValueError, SyntaxError):
                # print(f"Warning: Could not parse labels string: {label_str}. Setting labels to empty list.")
                label_list = [] # Handle parsing errors gracefully, maybe set to empty list

            # Create a multi-hot vector (list of 0s and 1s) for each example
            labels_binary = [1 if label in label_list else 0 for label in label_names]
            return {'labels': [float(label) for label in labels_binary]}

        raw_dataset = raw_dataset.map(map_labels_multilabel)
        print(f"raw_dataset: {raw_dataset["train"]["labels"]}")

    for row in raw_dataset['train']:
        print(f"Text: {row['text']}, Label: {row['labels']}")
    split_raw_dataset = split_dataset(raw_dataset)
    # for row in split_raw_dataset['train']:
    #     print(f"Text: {row['text']}, Label: {row['labels']}")
    tokenized_dataset = tokenize_dataset(split_raw_dataset, tokenizer)
    # for row in tokenized_dataset['train']:
    #     print(f"Labels: {row['labels']}, input_ids: {row['input_ids']}, attention_mask: {row['attention_mask']}")
    return tokenized_dataset, label_names, tokenizer

if __name__ == "__main__":
    tokenized_dataset, label_names = load_and_preprocess_dataset(model_id, task_type)
    # print(tokenized_dataset["train"].features.keys())
