from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("ReadCSV2").getOrCreate()

# Read the CSV file
df = spark.read.csv("transposed.csv", header=True, inferSchema=True)

df = df.repartition(3)

# print contents of each partition
partition_data = df.rdd.glom().collect()
for i, partition in enumerate(partition_data):
    print("Partition {}: {}".format(i, partition))



spark.stop()
