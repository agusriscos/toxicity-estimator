from os.path import join
import pandas as pd
import transformers
from src.finetune_transformer import CustomDataset, compute_metrics

if __name__ == '__main__':
    data_dirpath = "/home/agusriscos/toxicity-estimator/data/prep"
    logs_dirpath = "../training/bert"
    model_weights_path = join(logs_dirpath, "checkpoint-4500")

    test_df = pd.read_csv(join(data_dirpath, 'test_ruddit.csv'))
    test_texts = test_df["text"].values.tolist()
    test_labels = test_df["score"].values.tolist()

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = CustomDataset(test_encodings, test_labels)

    args = transformers.TrainingArguments(
        output_dir='../../training/dummy',
        logging_dir=join(logs_dirpath, 'test-logs')
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_weights_path)
    trainer = transformers.Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    result = trainer.predict(test_dataset)
    test_df["predict_score"] = result.predictions
    test_df.to_csv(join(data_dirpath, "predictions.csv"))
    print(result.metrics)