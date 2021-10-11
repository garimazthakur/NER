from __future__ import print_function
import argparse
import matplotlib
import os
import html
import warnings
from flask import Flask
from flask import jsonify
from flask import request
from utils import *
from transformers import AutoModelForTokenClassification
from config import MODEL_PTH


warnings.filterwarnings('ignore')
matplotlib.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
app.config['DEBUG'] = False


model = AutoModelForTokenClassification.from_pretrained(MODEL_PTH, local_files_only=True)


@app.route('/nertagging', methods=['POST'])

def tagEntity1():
	text = request.form['data']
	text = text.split("\n")
	text = [line.split() for line in text]
	text = [word for word in text if len(word) >0]
	# lan = request.form['lang']
	# text = html.unescape(text)
	print(text)
	y_pred = model.predict(text)
	preds_list, out_label_list = align_predictions(y_pred.predictions, y_pred.label_ids)
	test_x = []
	predict = []
	for line in text:
		test_x.extend(line[0])
	for pred in preds_list:
		predict.extend(pred)

	print(list(zip(test_x, predict)))

	return "done"
	# return jsonify(test)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5009, threaded=True)
