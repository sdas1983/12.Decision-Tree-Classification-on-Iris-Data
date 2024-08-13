# Decision Tree Classification on Iris Data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Display all columns
pd.set_option('display.max_columns', None)

# Load the Iris dataset
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
df.head()

# Encode species labels to numerical values
df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Define features and target variable
X = df.drop(columns=['species'])
y = df['species']

# Decision Tree with Entropy
classifier_entropy = DecisionTreeClassifier(criterion='entropy')
classifier_entropy.fit(X, y)

# Plot the decision tree with Entropy criterion
plt.figure(figsize=(15, 12))
plot_tree(classifier_entropy, filled=True, feature_names=X.columns, class_names=['Setosa', 'Versicolor', 'Virginica'], rounded=True)
plt.title('Decision Tree using Entropy')
plt.show()

# Decision Tree with Gini Index
classifier_gini = DecisionTreeClassifier(criterion='gini')
classifier_gini.fit(X, y)

# Plot the decision tree with Gini Index criterion
plt.figure(figsize=(15, 12))
plot_tree(classifier_gini, filled=True, feature_names=X.columns, class_names=['Setosa', 'Versicolor', 'Virginica'], rounded=True)
plt.title('Decision Tree using Gini Index')
plt.show()

# Hyperparameter Tuning and Pre-pruning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=30)

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [2, 4, 6, 8, 10, 12],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2]
}

# Initialize the Decision Tree Classifier and GridSearchCV
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Predict on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)
