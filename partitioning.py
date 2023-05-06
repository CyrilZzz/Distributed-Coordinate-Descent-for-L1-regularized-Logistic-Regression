from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, udf, array
from pyspark.ml.linalg import Vectors
import numpy as np

# Create a SparkSession
spark = SparkSession.builder.appName("distributed_lasso").getOrCreate()

# Read the CSV file
df = spark.read.csv("processed_dataset.csv", header=True, inferSchema=True)

# We extract the vector of labels which will be used in each partition
y = Vectors.dense(df.tail(1)[0][:-1])

# We remove it (last row) from the dataframe before partitioning it
df = df.filter(expr(f"feature_id < {df.count()-1}"))

def sigmoid(x, beta):
    dot_product = sum(xi * bi for xi, bi in zip(x, beta))
    return 1 / (1 + np.exp(-dot_product))

# Initialize the weight vector
beta = np.zeros(df.count())

observation_ids = df.columns
observation_ids.remove('feature_id')

x = [Vectors.dense([r[0] for r in df.select(observation_id).collect()]) for observation_id in observation_ids]

# Apply sigmoid function to each column and store results in a new vector
p= Vectors.dense([sigmoid(x_i,beta) for x_i in x])

w = p*(1-p)
z = ((y+1)/2 - p)/w

df = df.repartition(5)


# print contents of each partition
# partition_data = df.rdd.glom().collect()
# for i, partition in enumerate(partition_data):
#     print("Partition {}: {}".format(i, partition))

def compute_partition_avg(iterator):
    partition_sum = 0
    partition_count = 0
    for row in iterator:
        partition_sum += sum(row)
        partition_count += len(row)
    partition_avg = partition_sum / partition_count
    print("Partition avg: {}".format(partition_avg))
    yield partition_avg


# Apply the compute_partition_avg function to each partition and collect the results
partition_avgs = df.rdd.mapPartitions(compute_partition_avg).collect()

# Print the overall average value
overall_avg = sum(partition_avgs) / len(partition_avgs)
print("Overall avg: {}".format(overall_avg))


spark.stop()
