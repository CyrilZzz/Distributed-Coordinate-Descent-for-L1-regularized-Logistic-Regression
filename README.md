# Distributed Coordinate Descent for L1-regularized Logistic Regression

The DistributedLassoLogReg class implements a distributed version of Lasso Logistic Regression using PySpark, a distributed computing framework for big data processing. The objective of the algorithm is to train a model for binary classification that predicts the probability of a binary response variable (target variable) given a set of input variables (features). The Lasso regularization is used to select a subset of relevant features and to avoid overfitting.

The code performs the following steps:

Data preprocessing: the input dataset is cleaned and transformed into a format that can be used for training the model. The data is split into training and testing sets, and a binary encoding is applied to the target variable.

Model training: the data is partitioned and processed in parallel across multiple machines using PySpark. The algorithm uses the coordinate descent optimization method to update the model weights iteratively. In each iteration, the gradient of the objective function is computed using the training data from each partition, and the weight updates are aggregated across all partitions. The algorithm stops when a convergence criterion is met or the maximum number of iterations is reached.

Model testing and evaluation: the trained model is used to make predictions on the test set. The accuracy of the model is computed as the ratio of correct predictions to total predictions.

The code is designed to be run on a distributed computing cluster using PySpark. It requires a SparkSession to be created and a CSV file containing the training and testing data to be provided as input. The code can be modified to change the Lasso regularization parameter and the maximum number of iterations.
