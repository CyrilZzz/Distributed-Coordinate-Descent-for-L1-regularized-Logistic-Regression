# We want to transpose the initial dataset to have features as rows
#  for partitioning using Spark

import pandas as pd

data = pd.read_csv('small_dataset.csv').dropna()

data = data.transpose()

data.to_csv('processed_dataset.csv', index=False)

data['feature_id'] = [i for i in range(len(data))]

print(data)
