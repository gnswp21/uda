from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

class BERT_classifier():
    def __init__(self, model_cfgs):
        num_labels = model_cfgs.num_labels

        config = AutoConfig.from_pretrained(
            model_cfgs.model_name_or_path,
            num_labels=num_labels
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfgs.model_name_or_path,
            config=config
        )


