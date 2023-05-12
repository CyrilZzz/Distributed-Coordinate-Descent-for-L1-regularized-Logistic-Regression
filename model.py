from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.linalg import Vectors
import numpy as np
from optimization import coordinate_descent, line_search, objective_function
import pandas as pd
from sklearn.model_selection import train_test_split
import time


class DistributedLassoLogReg:
    def __init__(self, n_partitions, lmbd=1, max_iter=500):
        self.lmbd = lmbd
        self.n_partitions = n_partitions
        self.max_iter = max_iter
        self.beta = None

    @staticmethod
    def preprocessing(data, label_column):
        data.dropna(inplace=True)
        data.insert(0, 'constant', [1]*len(data))
        if 0 in data[label_column].unique():
            data[label_column] = data[label_column].apply(lambda x: 2 * x - 1)
        if '.' in label_column:  # There is an issue to extract the values of a column that includes a . in its name
            data.rename(columns={label_column : label_column.replace('.', '_')}, inplace=True)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        train_data = train_data.transpose()
        train_data['feature_id'] = [i for i in range(len(train_data))]

        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)


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

        cols = df.select(*observation_ids).collect()
        x = [Vectors.dense([r[i] for r in cols]) for i in range(len(observation_ids))]

        df = df.repartition(self.n_partitions)

        for iter in range(self.max_iter):

            # Apply sigmoid function to each column and store results in a new vector
            p = Vectors.dense([self.sigmoid(x_i, self.beta) for x_i in x])

            w = p * (1-p)
            z = ((y+1) / 2 - p) / w

            # Apply the coordinate_descent function to each partition and sum the results
            delta_beta = sum(df.rdd.mapPartitions(lambda partition: coordinate_descent(
                partition, x, w, z, self.beta, self.lmbd)).collect())

            alpha = line_search(x, y, 0.01, self.beta, delta_beta, 0, self.lmbd, 0.01, 0.5)
            if abs(objective_function(x,y,self.beta + alpha*delta_beta, self.lmbd) \
                    / objective_function(x, y, self.beta, self.lmbd) - 1) < 10 ** -4:
                self.beta = self.beta + alpha * delta_beta
                break
            self.beta = self.beta + alpha * delta_beta

    def predict(self, test_file):
        spark = SparkSession.builder.appName("testing").getOrCreate()
        # Read the CSV file
        test_data = spark.read.csv(test_file, header=True, inferSchema=True)

        y_test = [row[test_data.columns[-1]] for row in test_data.select(test_data.columns[-1]).collect()]

        test_data = test_data.drop(test_data.columns[-1])
        rows = test_data.collect()
        x_test = [np.array(row) for row in rows]

        y_pred = self.predict_internal(x_test)
        print("Accuracy on test set:", self.accuracy(y_pred, y_test))

        return y_pred

    def predict_internal(self, x):
        probas = np.array([self.sigmoid(x_i, self.beta) for x_i in x])
        return np.where(probas > 0.5, 1, -1)

    @staticmethod
    def sigmoid(x_i, beta):
        return 1 / (1 + np.exp(- np.dot(x_i, beta)))

    @staticmethod
    def accuracy(y_pred, y_test):
        return sum(y_pred == y_test)/len(y_pred)


#data = pd.read_csv('small_dataset.csv')

for i in range(1, 17, 3):
    model = DistributedLassoLogReg(i)
    #model.preprocessing(data, 'TenYearCHD')
    start_time = time.time()
    model.fit('train_data.csv')
    end_time = time.time()
    print(f"Elapsed time ({i}): {end_time-start_time} seconds")
    #print(model.beta)
    model.predict('test_data.csv')
