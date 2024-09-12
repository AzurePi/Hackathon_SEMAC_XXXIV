# %%
from fast_bert.data_cls import BertDataBunch
from fast_bert.prediction import BertClassificationPredictor

from pathlib import Path

import torch

import numpy as np

from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import pandas as pd
# %%

data = pd.read_csv('train.csv')

data = data.dropna()
data['ID'] = data['ID'].astype(int)
data['feeling'] = data['feeling'].astype(int)

porcentagem_val = 0.2

data_shuffled = data.sample(frac=1, random_state=42)

n_val = int(len(data) * porcentagem_val)

val_data = data_shuffled[:n_val]
train_data = data_shuffled[n_val:]

val_data.to_csv('val.csv', index=False)
train_data.to_csv('train.csv', index=False)
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
                          tokenizer='xlnet-base-cased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='feeling',
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=False,
                          model_type='xlnet-base-cased')

# %%

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
    databunch,
    pretrained_path='xlnet-base-cased',
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
with open("./learning_rate.txt", "w") as file:
    file.write(learner.learning_rate.__str__())

# %%

learner.fit(epochs=100,
            lr=learner.learning_rate,
            validate=True,  # Evaluate the model after each epoch
            schedule_type="linear",
            optimizer_type="lamb")

# %%

learner.save_model()

# %%
# device_cuda = torch.device("cuda")
#
# MODEL_PATH = OUTPUT_DIR / 'model_out'
#
# predictor = BertClassificationPredictor(
#     model_path=MODEL_PATH,
#     label_path=LABEL_PATH,  # location for labels.csv file
#     multi_label=False,
#     model_type='bert-base-cased',
#     do_lower_case=False,
#     device=device_cuda)  # set custom torch.device, defaults to cuda if available

# Batch predictions

# %%

reader = pd.read_csv("val.csv").sort_values(by="ID")

test_x = [str(x) for x in reader['text'].array]
test_ids = [int(x) for x in reader['ID'].array]

multiple_predictions = learner.predict_batch(test_x)

# with open("./teste.txt", "w") as file:
#     file.write(multiple_predictions.__str__())
with open(f"xlnet/predictions.csv", "w") as file:
    file.write("ID,feeling\n")
    for test_id, prediction in zip(test_ids, multiple_predictions):
        accu_0, accu_1 = prediction
        if accu_0 > accu_1:
            feeling = 0
        else:
            feeling = 1
        file.write(f"{int(test_id)},{feeling}\n")

# %%
predictions = pd.read_csv("xlnet/predictions.csv")
merged = pd.merge(predictions, reader, on="ID", suffixes=('_pred', '_orig'))
merged['match'] = merged['feeling_pred'] == merged['feeling_orig']
acuracia = merged['match'].mean() * 100
print(acuracia)
