import pyspark

# Load data from CSV file
data = pyspark.read.csv("small_dataset.csv", header=True, inferSchema=True)

print(data)

# Partition data based on features into 100 partitions
partition_cols = ["col1", "col2", ...] # List of feature column names to partition by
data = data.repartitionByCol(partition_cols, numPartitions=100)

# Split data into training and test sets
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Prepare features and label columns
features = [col for col in train.columns if col != "label"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
train = assembler.transform(train).select("features", "label")
test = assembler.transform(test).select("features", "label")

# Configure d-GLMNET with Lasso regularization
lr = LogisticRegression(regParam=0.1, elasticNetParam=1.0)

# Fit the model on the training data
model = lr.fit(train)

# Evaluate the model on the test data
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(predictions)
print("AUC = ", auc)
