import os
from conll_to_boi import conll_to_boi
from config import *
from model import model_init
from dataset import NERDataset, Dataset
from collections import Counter
from utils import *
import numpy as np
import torch
import logging
from transformers import PrinterCallback, EarlyStoppingCallback
from transformers import Trainer
import warnings
import json

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

all_datasets = []
if not os.path.isfile(TRN_PTH):
	print("\n>>> no  BOI format data found.\n>>> creating BOI format data.")
	conll_to_boi()
data_train = read_ANERcorp(TRN_PTH)
data_test = read_ANERcorp(TST_PTH)


print(f">>> {Counter([label for sentence in data_test for label in sentence[1]])}")
print(f">>> {Counter([label for sentence in data_train for label in sentence[1]])}")
label_list = list(Counter([label for sentence in data_test for label in sentence[1]]).keys())
print(f">>> {label_list}")

data_AJGT = Dataset("ANERCorp", data_train, data_test, label_list)
all_datasets.append(data_AJGT)

for x in all_datasets:
	print(f">>> using {x.name} dataset")

for d in all_datasets:
	if d.name == DATASET_NAME:
		selected_dataset = d
		print('Dataset found')
		break

label_map = {v: index for index, v in enumerate(selected_dataset.label_list)}
print(label_map)

train_dataset = NERDataset(
	texts=[x[0] for x in selected_dataset.train],
	tags=[x[1] for x in selected_dataset.train],
	label_list=selected_dataset.label_list,
	model_name=MODEL_NAME,
	max_length=256
)

test_dataset = NERDataset(
	texts=[x[0] for x in selected_dataset.test],
	tags=[x[1] for x in selected_dataset.test],
	label_list=selected_dataset.label_list,
	model_name=MODEL_NAME,
	max_length=256
)
# global inv_label_map
inv_label_map = {i: label for i, label in enumerate(label_list)}
with open(os.path.join(RES_PTH, 'inv_label_map.json'), 'w', encoding ='utf8') as json_file:
	json.dump(inv_label_map, json_file, ensure_ascii=False)



steps_per_epoch = (len(selected_dataset.train) // (
		training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
total_steps = steps_per_epoch * training_args.num_train_epochs
print(f">>> steps per epoch: {steps_per_epoch}")
print(f">>> total steps for training: {total_steps}")

training_args.warmup_steps = total_steps * warmup_ratio

trainer = Trainer(
	model=model_init(MODEL_NAME, label_map),
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	compute_metrics=compute_metrics,
	callbacks=[PrinterCallback, EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
metrics = trainer.evaluate(metric_key_prefix="eval",)

# print(metrics)
trainer.save_model(MODEL_PTH)
