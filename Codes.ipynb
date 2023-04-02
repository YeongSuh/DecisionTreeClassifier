# 1) Import Libraries

import pandas as pd
import numpy as np
import sklearn

# 2) File upload

from google.colab import files
myfile = files.upload()

# 3) Load the data

data_logit = pd.read_csv("data_logit.csv", index_col=0)
data_logit

# 4) Split the data with train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_logit.drop(['y'], axis=1), data_logit['y'], train_size=0.7, random_state=42)

# 5) Specify the model

from sklearn.tree import DecisionTreeClassifier
dt_fit = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)
dt_fit.fit(X_train, y_train)

# 6) Print the Confusion Matrix

print("\nDecision Tree - Train Confusion Matrix\n\n", pd.crosstab(y_train, dt_fit.predict(X_train), rownames=['Acutal'], colnames=['Predicted']))

# 7) Evaluate the result with accuracy score and classification report
# Accuracy Score

from sklearn.metrics import accuracy_score, classification_report
print("\nDecision Tree - Train accuracy\n\n", round(accuracy_score(y_train, dt_fit.predict(X_train)), 3))

# Classification Report

print("\nDecision Tree - Train Classification Report\n", classification_report(y_train, dt_fit.predict(X_train)))

# 8) Print the result of the tree model

from sklearn import tree
text_representation = tree.export_text(dt_fit)
print(text_representation)

# 9) Draw graphs

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (25,20))
_ = tree.plot_tree(dt_fit, 
                   feature_names= data_logit.columns.drop(['y']),
                   class_names = list(set(np.array(data_logit.y).astype('<U10'))))
