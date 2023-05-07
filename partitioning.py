from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.linalg import Vectors
import numpy as np
from optimization import coordinate_descent

# Create a SparkSession
spark = SparkSession.builder.appName("distributed_lasso").getOrCreate()

# Choose penalty level
lmbd = 0.2

# Read the CSV file
df = spark.read.csv("processed_dataset.csv", header=True, inferSchema=True)

# We extract the vector of labels which will be used in each partition
y = Vectors.dense(df.tail(1)[0][:-1])

# We remove it (last row) from the dataframe before partitioning it
df = df.filter(expr(f"feature_id < {df.count()-1}"))

# Initialize the weight vector
beta = np.zeros(df.count())

observation_ids = df.columns
observation_ids.remove('feature_id')

x = [Vectors.dense([r[0] for r in df.select(observation_id).collect()]) for observation_id in observation_ids]

df = df.repartition(5)

def sigmoid(x, beta):
    dot_product = sum(xi * bi for xi, bi in zip(x, beta))
    return 1 / (1 + np.exp(-dot_product))


nb_iter = 1500  # fixed number of iterations (for testing)

for iter in range(nb_iter):
  
    # Apply sigmoid function to each column and store results in a new vector
    p= Vectors.dense([sigmoid(x_i,beta) for x_i in x])

    w = p*(1-p)
    z = ((y+1)/2 - p)/w


    # print contents of each partition
    # partition_data = df.rdd.glom().collect()
    # for i, partition in enumerate(partition_data):
    #     print("Partition {}: {}".format(i, partition))

    # Apply the coordinate_descent function to each partition and sum the results
    delta_beta = sum(df.rdd.mapPartitions(lambda partition: coordinate_descent(partition, x, w, z, beta, lmbd)).collect())

    alpha = 0.3
    beta = beta + alpha*delta_beta

print(beta)

def predict(x,beta):
    probas = np.array([sigmoid(x_i,beta) for x_i in x])
    return np.where(probas>0.5,1,0)

def accuracy(y_pred,y_test):
    return sum(y_pred==y_test)/len(y_pred)

# Accuracy on training set (waiting for split)
y_pred = predict(x,beta)

print(accuracy(y_pred,y))


spark.stop()
