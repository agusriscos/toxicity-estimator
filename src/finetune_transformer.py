from os.path import join
import pandas as pd

from sklearn.metrics import mean_squared_error

import torch
from torch.utils.data import Dataset
import torch.nn as nn

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


class PosModel(nn.Module):
    def __init__(self):
        super(PosModel, self).__init__()

        self.base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 1)  # output features from bert is 768 and 2 is ur number of labels

    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        # You write you new head here
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)

        return outputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    r_mse = mean_squared_error(labels, predictions, squared=False)
    return {"r_mse": r_mse}


def get_hugging_datasets(train, val, model_name="distilbert-base-uncased"):
    train_texts = train["text"].values.tolist()
    train_labels = train["score"].values.tolist()
    val_texts = val["text"].values.tolist()
    val_labels = val["score"].values.tolist()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    return train_dataset, val_dataset


def main():
    data_dirpath = "/home/agusriscos/toxicity-estimator/data/prep"
    training_dirpath = "/home/agusriscos/toxicity-estimator/training/bert"
    train_df = pd.read_csv(join(data_dirpath, "train_ruddit.csv"))
    val_df = pd.read_csv(join(data_dirpath, "test_ruddit.csv"))

    train_dataset, val_dataset = get_hugging_datasets(train_df, val_df,
                                                      "distilbert-base-uncased")

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True
    )
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
