import json
from datasets import load_dataset
from transformers import AutoTokenizer

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

def load_raw_dataset(train_file, test_file, valid_file):
    dataset = load_dataset(dataset_id)
    dataset["train"] = dataset["train"].select(range(1000))
    return dataset

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = MAX_TOKEN_LENGTH
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, return_tensors="pt")
    return dataset.map(tokenize, batched=True, remove_columns=["text"])

def load_and_preprocess_dataset(train_file, test_file, valid_file, model_id, task_type="binary"):
    raw_dataset = load_raw_dataset(train_file, test_file, valid_file)
    tokenizer = get_tokenizer(model_id)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    return tokenized_dataset

if __name__ == "__main__":
    tokenized_dataset = load_and_preprocess_dataset(train_file, test_file, valid_file, model_id, task_type)
    print(tokenized_dataset["train"].features.keys())
