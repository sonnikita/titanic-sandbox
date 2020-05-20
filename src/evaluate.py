import pandas as pd
import json
import os
import pickle
import sys

from sklearn.metrics import roc_auc_score, accuracy_score

if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython evaluate.py model_file data_valid metrics_file\n')
    sys.exit(1)

model_file = sys.argv[1]
valid_file = sys.argv[2]
auc_file = sys.argv[3]
acc_file = sys.argv[4]

with open(model_file, 'rb') as fd:
    model = pickle.load(fd)

valid_df = pd.read_csv(valid_file)

used_cols = [
    'Pclass', 'Age', 'is_fem'
]

target = ['Survived', ]

preds = model.predict(valid_df[used_cols])
preds_proba = model.predict_proba(valid_df[used_cols])[:,1]
auc_score = roc_auc_score(valid_df[target[0]], preds_proba)
acc_score = accuracy_score(valid_df[target[0]], preds)

with open(auc_file, 'w') as fd:
    fd.write('{:4f}\n'.format(auc_score))
    #fd.write('%s\n' % json.dumps({
    #    'auc': auc_score,
    #    'accuracy': acc_score,
    #}, ensure_ascii=False))

with open(acc_file, 'w') as fd:
    fd.write('{:4f}\n'.format(acc_score))
