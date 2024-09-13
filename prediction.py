import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


test_df = pd.read_csv("./data/test.csv")

tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model")

test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)

test_dataset = TestDataset(test_encodings)

predict_trainer = Trainer(model=model)
predictions = predict_trainer.predict(test_dataset)

predicted_labels = [x.argmax() for x in predictions.predictions]
test_df["feeling"] = predicted_labels

test_df.to_csv("./predictions.csv", index=False, columns=['ID', 'feeling'])
