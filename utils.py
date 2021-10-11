from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from config import RES_PTH
import pandas as pd
import numpy as np
import torch
import json
import os


def read_ANERcorp(path):
	df = pd.read_csv(path, encoding="utf-8")
	data = df.groupby(['file', 'sent_num']).agg(list)
	data = [tuple(x) for x in data.to_numpy()]

	return data


def align_predictions(predictions, label_ids):  # , inv_label_map

	with open(os.path.join(RES_PTH, 'inv_label_map.json')) as f:
		inv_label_map = json.load(f)
	preds = np.argmax(predictions, axis=2)

	batch_size, seq_len = preds.shape

	out_label_list = [[] for _ in range(batch_size)]
	preds_list = [[] for _ in range(batch_size)]

	for i in range(batch_size):
		for j in range(seq_len):
			if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
				out_label_list[i].append(inv_label_map[str(label_ids[i][j])])
				preds_list[i].append(inv_label_map[str(preds[i][j])])

	return preds_list, out_label_list


def compute_metrics(p):
	preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
	print(f"\n{classification_report(out_label_list, preds_list, digits=4)}")
	return {
		"accuracy_score": accuracy_score(out_label_list, preds_list),
		"precision": precision_score(out_label_list, preds_list),
		"recall": recall_score(out_label_list, preds_list),
		"f1": f1_score(out_label_list, preds_list),
	}
