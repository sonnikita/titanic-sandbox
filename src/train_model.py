import pandas as pd
import os
import pickle
import sys

from sklearn.ensemble import RandomForestClassifier

if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython prepare_data.py data_train\n')
    sys.exit(1)

train_file = sys.argv[1]

train_df = pd.read_csv(train_file)

used_cols = [
    'Pclass', 'Age', 'is_fem'
]

target = ['Survived', ]

model = RandomForestClassifier(n_estimators=2400)
model.fit(train_df[used_cols], train_df[target[0]])

with open('model.pickle', 'wb') as model_file:
	pickle.dump(model, model_file)
