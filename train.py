import json
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from src.dataset import load_and_preprocess_dataset
from huggingface_hub import HfFolder
from src.eval import compute_metrics

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

def train_model(model_id, task_type):

    tokenized_dataset, label_names, tokenizer = load_and_preprocess_dataset(model_id, task_type)

    num_labels = len(label_names)
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for i, label in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained( # Load classification model
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir= "results/ModernBERT-domain-classifier",
        overwrite_output_dir=True, # Overwrite if checkpoint exists
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate = 5e-5,
        num_train_epochs = 1,
        bf16=False, 
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub = False,
        hub_strategy="every_save",
        hub_token=HfFolder.get_token(), # Ensure you have HF token setup if pushing to hub
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    train_model(config["base_model"], config["task_type"])