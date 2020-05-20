import pandas as pd
import os
import sys

if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython prepare_data.py data_train\n')
    sys.exit(1)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

train_file = sys.argv[1]
output_train = os.path.join('data', 'prepared', 'train.csv')
output_test = os.path.join('data', 'prepared', 'valid.csv')

train_df = pd.read_csv(train_file)

train_df['is_fem'] = train_df.Sex.apply(
    lambda x: 1 if x == 'female' else 0
)

used_cols = [
    'Pclass', 'Age', 'is_fem'
]

target = ['Survived', ]

mkdir_p(os.path.join('data', 'prepared'))

train_df = train_df[used_cols + target]
train_df.dropna(inplace=True)

train_test_rate = 0.75
l = int(train_test_rate * train_df.shape[0])

train_df.iloc[:l].to_csv(output_train, index=False)
train_df.iloc[l:].to_csv(output_test, index=False)
