# We want to transpose the initial dataset to have features as rows
#  for partitioning using Spark

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('small_dataset.csv')


def preprocessing(data, label_column):

    data.dropna(inplace=True)

    data.insert(0, 'constant', [1 for i in range(len(data))])

    data[label_column] = data[label_column].apply(lambda x: 2*x - 1)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_data = train_data.transpose()

    train_data['feature_id'] = [i for i in range(len(train_data))]

    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    return train_data, test_data

print(preprocessing(data, 'TenYearCHD')[0])
