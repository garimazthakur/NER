from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import re
from config import ANN_PTH, TRN_PTH, TST_PTH


def conll_to_boi():
	views = ("train", "test")
	for view in views:
		print(f"\n>>> converting {view}ing files")
		path = os.path.join(ANN_PTH, view)
		files = os.listdir(os.path.join(ANN_PTH, view))
		files = [(file, f"{file.replace('txt', 'ann')}") for file in files if '.txt' in file]
		df = pd.DataFrame()
		for iof in tqdm(files, ):
			try:
				with open(os.path.join(path, iof[0]), "r", encoding="utf-8", errors="strict") as f:
					text = f.read()
				idx_df = pd.DataFrame(
					[(ele.start(), ele.end() - 1) for ele in re.finditer(r'\S+', text)],
					columns=['start', 'end']
				)
				sent_df = pd.DataFrame(text.split('\n'), columns=['text'])
				sent_df['text'] = sent_df.text.str.strip().str.replace('\t', ' ')
				sent_df = sent_df[sent_df.text != ''].reset_index(drop=True)
				sent_df['sent_num'] = sent_df.index + 1
				sent_df['text'] = sent_df['text'].str.split()
				sent_df = sent_df.explode('text')
				sent_df = sent_df.reset_index(drop=True)
				sent_df = sent_df.merge(idx_df, right_index=True, left_index=True)
				ann_df = pd.read_csv(os.path.join(path, iof[1]), sep='\t', names=['enity_num', 'ner', 'text'])
				ann_df[['ner', 'start', 'end']] = ann_df['ner'].str.split().apply(pd.Series)
				ann_df['len'] = ann_df['text'].str.split().apply(len)
				sent_df = sent_df.merge(ann_df.astype({'start': int}), on='start', how='left')
				g = sent_df['ner'].notna().cumsum()
				sent_df['len'] = sent_df['len'].fillna(0).astype(int)
				f = lambda x: x['ner'].ffill(limit=1 if x['len'].iat[0] < 2 else x['len'].iat[0] - 1)
				sent_df['ner_2'] = sent_df.groupby(g, group_keys=False).apply(f)
				sent_df['ner_3'] = sent_df['ner_2'].shift(1)
				sent_df = sent_df.fillna({'ner_2': 'O', 'ner_3': 'O', 'ner': "O"})
				sent_df['boi'] = np.select(
					[
						(sent_df.ner == sent_df['ner_2']) & (sent_df.ner != sent_df.ner_3) & (sent_df.ner != 'O'),
						(sent_df.ner != sent_df['ner_2']) & (sent_df.ner_2 == sent_df.ner_3) & (
									sent_df.ner != sent_df.ner_3),
						(sent_df.ner == sent_df.ner_2) & (sent_df.ner_2 == sent_df.ner_3) & (sent_df.len == 1)
					],
					[
						'B',
						'I',
						'B'
					],
					'O'
				)
				sent_df['len_boi'] = sent_df['len'].replace(0, np.nan).ffill()
				sent_df['boi'] = np.where(
					(sent_df.len_boi == 1) &
					(sent_df.boi == 'I'),
					'O',
					sent_df.boi
				)
				sent_df['ner_2'] = np.where(
					(sent_df.ner_2 != 'O') & (sent_df.boi == 'O'),
					'O',
					sent_df.ner_2
				)

				sent_df['ner'] = np.select(
					[sent_df.ner_2 != 'O'],
					[sent_df['boi'] + '-' + sent_df['ner_2']],
					'O'
				)
				df = df.append(sent_df[['text_x', 'ner', 'sent_num']].rename(columns={'text_x': 'text'}))
				df = df.append(pd.DataFrame({'text': [np.nan], 'ner': [np.nan], 'sent_num': [np.nan]}))
			except Exception as ex:
				print(iof)
				print(ex)
			df['file'] = np.where(df.ner.isnull(), 1, np.nan)
			df['file'] = df.file.fillna(0).cumsum()
			if view == "train":
				df.to_csv(TRN_PTH, encoding="utf-8", index=False)
			else:
				df.to_csv(TST_PTH, encoding="utf-8", index=False)


if __name__ == "__main__":
	conll_to_boi()
