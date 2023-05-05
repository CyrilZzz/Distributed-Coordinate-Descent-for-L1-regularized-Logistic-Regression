from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import rand


spark = SparkSession.builder.appName("PartitionByFeatures").getOrCreate()

# Load data from CSV file
data = spark.read.csv("small_dataset.csv", header=True, inferSchema=True)

features = data.columns
features.remove('TenYearCHD')

# partition the DataFrame by features and write to disk
data.write.partitionBy('TenYearCHD','glucose').mode('overwrite').format('parquet').save("partitioned_data")

# read the partitioned data from disk
partitioned_data = spark.read.format('parquet').load("partitioned_data")

# get the number of partitions of the DataFrame
num_partitions = partitioned_data.rdd.getNumPartitions()

print("Number of partitions:", num_partitions)
