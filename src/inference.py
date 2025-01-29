from transformers import pipeline
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_id = "/Users/shrijayan.rajendran/projects/personal/monk/modern_bert-finetuning/results/ModernBERT-domain-classifier/checkpoint-11"  # Or "answerdotai/ModernBERT-large"
device = "mps" # Use "cuda" for GPU, or "cpu" for CPU

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device='mps'
)

input_text = "He walked to the sun."
results = pipe(input_text)
pprint(results)
