import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


df = pd.read_csv("train.csv")

df.dropna(inplace=True)

df = df[['text', 'feeling']]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['feeling'].tolist(),
    test_size=0.2
)

train_labels = [int(label) for label in train_labels]
val_labels = [int(label) for label in val_labels]

tokenizer = AutoTokenizer.from_pretrained("Ibrahim-Alam/finetuning-roberta-base-on-tweet_sentiment_binary")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=236)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=236)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    "Ibrahim-Alam/finetuning-roberta-base-on-tweet_sentiment_binary", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=9,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
eval_results = trainer.evaluate()
print(eval_results)

model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
