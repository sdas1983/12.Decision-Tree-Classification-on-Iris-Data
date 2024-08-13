# Decision-Tree-Classification-on-Iris-Data

#Overview
This project demonstrates the application of Decision Tree Classification on the well-known Iris dataset. The objective is to classify the species of Iris flowers based on features like sepal length, sepal width, petal length, and petal width. The dataset is a classic example in machine learning and statistics, often used for testing algorithms.

#Dataset
The Iris dataset contains 150 samples of iris flowers, with three species: Iris-setosa, Iris-versicolor, and Iris-virginica. Each species has 50 samples, and the dataset includes the following features:

Sepal Length: The length of the sepal in centimeters.
Sepal Width: The width of the sepal in centimeters.
Petal Length: The length of the petal in centimeters.
Petal Width: The width of the petal in centimeters.
Species: The species of the iris flower (setosa, versicolor, virginica).

#Project Structure
iris_classification.ipynb: The Jupyter notebook containing the full implementation of the Decision Tree Classification, including hyperparameter tuning and model evaluation.
iris_classification.py: The Python script version of the notebook, useful for running in a non-notebook environment.
README.md: This file, providing an overview of the project and dataset.
plots/: Directory containing visualizations and decision tree plots generated during the analysis.

#Methodology
##1. Data Loading and Preprocessing
Loaded the Iris dataset using Seaborn's built-in datasets.
Encoded the species labels to numerical values for model training.
##2. Decision Tree Classification
Built and trained decision tree models using both the Entropy and Gini Index criteria.
Visualized the decision trees to understand the decision-making process of the model.
##3. Hyperparameter Tuning
Performed hyperparameter tuning using GridSearchCV to find the best parameters for the decision tree model.
Evaluated the tuned model's performance using accuracy, confusion matrix, and classification report.
##4. Results
The best model parameters were identified, and the model achieved a high level of accuracy in classifying the Iris species.
Visualizations and confusion matrix were generated to provide insights into model performance.
