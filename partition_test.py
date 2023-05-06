from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# create a SparkSession
spark = SparkSession.builder.appName("example").getOrCreate()

# define the schema
schema = StructType([
    StructField("col1", IntegerType(), True),
    StructField("col2", IntegerType(), True),
    StructField("col3", IntegerType(), True)
])

# create the data
data = [
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (13, 14, 15),
    (16, 17, 18)
]

# create the DataFrame
df = spark.createDataFrame(data, schema)

# show the DataFrame
df.show()

spark.stop()
