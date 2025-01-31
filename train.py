import json
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score
from src.dataset import load_and_preprocess_dataset
from src.model import load_model
from huggingface_hub import HfFolder

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

def compute_metrics(eval_pred):
    """
    Computes F1 score for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"f1": f1_score(labels, predictions, average="weighted")}

def train_model(train_file, test_file, valid_file, model_id, task_type):
    """
    Full pipeline for training.
    """
    
    tokenized_dataset = load_and_preprocess_dataset(train_file, test_file, valid_file, model_id, task_type)
    
    # Prepare model labels - useful for inference
    labels = tokenized_dataset["train"].features["label"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    
    model = load_model(model_id, num_labels, label2id, id2label)

    training_args = TrainingArguments(
        output_dir= "modernbert-llm-router",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
            num_train_epochs=5,
        bf16=True, # bfloat16 training
        optim="adamw_torch_fused", # improved optimizer
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # push to hub parameters
        hub_strategy="every_save",
        hub_token=HfFolder.get_token(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    train_model(config['train_file'], config['test_file'], config['valid_file'], config["base_model"], config["task_type"])
