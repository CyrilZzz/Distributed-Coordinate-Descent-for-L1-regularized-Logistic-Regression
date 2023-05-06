from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("distributed_lasso").getOrCreate()

# Read the CSV file
df = spark.read.csv("processed_dataset.csv", header=True, inferSchema=True)

df.show()

df = df.repartition(10)


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
