# We want to transpose the initial dataset to have features as rows
#  for partitioning using Spark

import pandas as pd

data = pd.read_csv('small_dataset.csv').dropna().head(300)

data = data.transpose()

data['feature_id'] = [i for i in range(len(data))]

data.to_csv('processed_dataset.csv', index=False)

print(data)
