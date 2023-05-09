from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.linalg import Vectors
import numpy as np
from optimization import coordinate_descent, line_search
import pandas as pd
from sklearn.model_selection import train_test_split


class Distributed_Lasso_LogReg:
    def __init__(self, n_partitions, lmbd=1):
        self.lmbd = lmbd
        self.n_partitions = n_partitions
        self.beta = None

    def preprocessing(self, data, label_column):
        data.dropna(inplace=True)
        data.insert(0, 'constant', [1 for i in range(len(data))])
        data[label_column] = data[label_column].apply(lambda x: 2*x - 1)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        train_data = train_data.transpose()
        train_data['feature_id'] = [i for i in range(len(train_data))]

        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)

        return train_data, test_data

    def fit(self, train_file):
        spark = SparkSession.builder.appName("training").getOrCreate()
        # Read the CSV file
        df = spark.read.csv(train_file, header=True, inferSchema=True)

        # We extract the vector of labels which will be used in each partition
        y = Vectors.dense(df.tail(1)[0][:-1])

        # We remove it (last row) from the dataframe before partitioning it
        df = df.filter(expr(f"feature_id < {df.count()-1}"))

        # Initialize the weight vector
        self.beta = np.zeros(df.count())

        observation_ids = df.columns
        observation_ids.remove('feature_id')

        x = [Vectors.dense([r[0] for r in df.select(observation_id).collect()]) for observation_id in observation_ids]

        df = df.repartition(self.n_partitions)

        nb_iter = 10  # fixed number of iterations (for testing)

        for iter in range(nb_iter):

            # Apply sigmoid function to each column and store results in a new vector
            p = Vectors.dense([self.sigmoid(x_i, self.beta) for x_i in x])

            w = p*(1-p)
            z = ((y+1)/2 - p)/w

            # Apply the coordinate_descent function to each partition and sum the results
            delta_beta = sum(df.rdd.mapPartitions(lambda partition: coordinate_descent(partition, x, w, z, self.beta, self.lmbd)).collect())

            alpha = line_search(x, y, 0.01, self.beta, delta_beta, 0, self.lmbd, 0.01, 0.5)
            self.beta = self.beta + alpha*delta_beta

    def predict(self, test_file):
        spark = SparkSession.builder.appName("testing").getOrCreate()
        # Read the CSV file
        test_data = spark.read.csv(test_file, header=True, inferSchema=True)

        y_test = [row.TenYearCHD for row in test_data.select('TenYearCHD').collect()]

        test_data = test_data.drop('TenYearCHD')
        rows = test_data.collect()
        x_test = [np.array(row) for row in rows]

        y_pred = self.predict_internal(x_test)
        print("Accuracy on test set:", self.accuracy(y_pred, y_test))

        return y_pred

    def predict_internal(self, x):
        probas = np.array([self.sigmoid(x_i, self.beta) for x_i in x])
        return np.where(probas > 0.5, 1, -1)

    @staticmethod
    def sigmoid(x, beta):
        dot_product = sum(xi * bi for xi, bi in zip(x, beta))
        return 1 / (1 + np.exp(-dot_product))

    @staticmethod
    def accuracy(y_pred, y_test):
        return sum(y_pred == y_test)/len(y_pred)

    def stop_spark(self):
        self.spark.stop()


data = pd.read_csv('small_dataset.csv')

model = Distributed_Lasso_LogReg(5)
train, test = model.preprocessing(data, 'TenYearCHD')

model.fit('train_data.csv')
model.predict('test_data.csv')
print(model.beta)
