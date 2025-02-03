import json
from transformers import Trainer, TrainingArguments
from src.dataset import load_and_preprocess_dataset
from src.model import load_model
from huggingface_hub import HfFolder
from src.eval import compute_metrics

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

def train_model(model_id, task_type):
    """
    Full pipeline for training.
    """
    
    tokenized_dataset, label_names = load_and_preprocess_dataset(model_id, task_type)
    
    labels = label_names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    
    model = load_model(model_id, num_labels, label2id, id2label)
    
    training_args = TrainingArguments(
        output_dir= "ModernBERT-domain-classifier",
        per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        learning_rate=5e-5,
        num_train_epochs=5,
        bf16=True, 
        optim="adamw_torch_fused", 
        logging_strategy="steps",
        logging_steps=100,
        # eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # load_best_model_at_end=True,
        metric_for_best_model="f1",
        hub_strategy="every_save",
        hub_token=HfFolder.get_token(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        # eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    train_model(config["base_model"], config["task_type"])
