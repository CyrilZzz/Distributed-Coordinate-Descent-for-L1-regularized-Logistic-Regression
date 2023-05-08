from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# load the data
df = pd.read_csv('small_dataset.csv').dropna()

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df['TenYearCHD'], test_size=0.2, random_state=42)

# train the Lasso logistic regression model
clf = LogisticRegression(penalty='l1', solver='liblinear', C=2, random_state=42)
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# evaluate the performance of the model
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)
print(clf.coef_)
