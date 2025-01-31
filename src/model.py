from transformers import AutoModelForSequenceClassification

def load_model(model_id, num_labels, label2id, id2label):
    """
    Loads a ModernBERT model with the correct classification head.

    Args:
        model_id (str): Hugging Face model ID.
        num_labels (int): Number of labels for classification.
    
    Returns:
        model (transformers.PreTrainedModel): ModernBERT model.
    """
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
    )
    
    return model
