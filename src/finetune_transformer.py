
from os.path import join
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
from torch.utils.data import Dataset
import transformers
transformers.logging.set_verbosity_error()


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def get_hugging_datasets(train, val, model_name="dccuchile/bert-base-spanish-wwm-cased"):
    train_texts = train["text"].values.tolist()
    train_labels = train["claim"].values.tolist()
    val_texts = val["text"].values.tolist()
    val_labels = val["claim"].values.tolist()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    return train_dataset, val_dataset


def main():
    data_dirpath = "/home/agusriscos/verifiable-phrase-detection/data"
    training_dirpath = "/home/agusriscos/verifiable-phrase-detection/training/bert"
    train_df = pd.read_csv(join(data_dirpath, "prep_train.csv"))
    val_df = pd.read_csv(join(data_dirpath, "raw_val.csv"))

    train_dataset, val_dataset = get_hugging_datasets(train_df, val_df, "dccuchile/bert-base-spanish-wwm-cased")

    model = transformers.AutoModelForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-cased",
                                                                            num_labels=2)
    training_args = transformers.TrainingArguments(output_dir=training_dirpath,
                                                   overwrite_output_dir=True, num_train_epochs=4, logging_steps=100,
                                                   evaluation_strategy="steps", eval_steps=100, learning_rate=2e-5,
                                                   per_device_train_batch_size=4, per_device_eval_batch_size=4,
                                                   save_steps=100)
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    main()