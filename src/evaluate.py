import pandas as pd
import os
import pickle
import sys

from sklearn.metrics import roc_auc_score

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython evaluate.py model_file data_valid metrics_file\n')
    sys.exit(1)

model_file = sys.argv[1]
valid_file = sys.argv[2]
metrics_file = sys.argv[3]

with open(model_file, 'rb') as fd:
    model = pickle.load(fd)

valid_df = pd.read_csv(valid_file)

used_cols = [
    'Pclass', 'Age', 'is_fem'
]

target = ['Survived', ]

preds = model.predict_proba(valid_df[used_cols])[:,1]
score = roc_auc_score(valid_df[target[0]], preds)

with open(metrics_file, 'w') as fd:
    fd.write('{:4f}\n'.format(score))