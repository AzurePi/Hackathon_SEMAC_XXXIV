# %%
from fast_bert.data_cls import BertDataBunch

from pathlib import Path

import torch

from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import pandas as pd

# %%

data = pd.read_csv('./data/train.csv')

data = data.dropna()
data['ID'] = data['ID'].astype(int)
data['feeling'] = data['feeling'].astype(int)

porcentagem_val = 0.2

data_shuffled = data.sample(frac=1, random_state=42)

n_val = int(len(data) * porcentagem_val)

val_data = data_shuffled[:n_val]
train_data = data_shuffled[n_val:]

val_data.to_csv('./data/val.csv', index=False)
train_data.to_csv('./data/train.csv', index=False)
print("ok")

# %%

model_type = ["roberta"]
DATA_PATH = Path('./')
LABEL_PATH = Path('./')
LOG_PATH = Path('./logs')
MODEL_PATH = Path('./lm_model_{}/'.format(model_type))
OUTPUT_DIR = Path('./output')

MODEL_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# %%

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='Ibrahim-Alam/finetuning-roberta-base-on-tweet_sentiment_binary',
                          train_file='./data/train.csv',
                          val_file='./data/val.csv',
                          label_file='./data/labels.csv',
                          text_col='text',
                          label_col='feeling',
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=False,
                          model_type='roberta')

# %%

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
    databunch,
    pretrained_path='Ibrahim-Alam/finetuning-roberta-base-on-tweet_sentiment_binary',
    metrics=metrics,
    device=device_cuda,
    logger=logger,
    output_dir=OUTPUT_DIR,
    finetuned_wgts_path=None,
    warmup_steps=500,
    multi_gpu=True,
    is_fp16=True,
    multi_label=False,
    logging_steps=50
)

# %%

learner.lr_find(start_lr=1e-5, optimizer_type='lamb')
print(learner.learning_rate)

# %%

learner.fit(epochs=100,
            lr=learner.learning_rate,
            validate=True,  # Evaluate the model after each epoch
            schedule_type="warmup_cosine",
            optimizer_type="lamb")

learner.save_model()

# %%

reader = pd.read_csv("./data/val.csv").sort_values(by="ID")

test_x = [str(x) for x in reader['text'].array]
test_ids = [int(x) for x in reader['ID'].array]

multiple_predictions = learner.predict_batch(test_x)

# %%

with open(f"./Ibrahim-Alam/predictions.csv", "w") as file:
    file.write("ID,feeling\n")
    for test_id, prediction in zip(test_ids, multiple_predictions):
        accu_0, accu_1 = prediction
        feeling = int(not accu_0 > accu_1)
        file.write(f"{test_id},{feeling}\n")

# %%

predictions = pd.read_csv("./Ibrahim-Alam/predictions.csv")

merged = pd.merge(predictions, reader, on="ID", suffixes=('_pred', '_orig'))
merged['match'] = merged['feeling_pred'] == merged['feeling_orig']

acuracia = merged["match"].mean() * 100

print(acuracia)
