# Machine-learning-basics-Day-1
Machine learning 

Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn from data and make predictions or decisions without being explicitly programmed. It involves developing algorithms that can recognize patterns, analyze data, and improve performance over time.

Key Components of Machine Learning

1. Data – The foundation of ML. Clean and structured data is essential for accurate predictions.


2. Features – Attributes or variables used to make predictions. Feature engineering improves model performance.


3. Algorithms – Mathematical models that learn from data. Examples include decision trees, neural networks, and support vector machines.


4. Model Training – The process of feeding data into an algorithm so it can learn patterns and relationships.


5. Evaluation – Measuring model performance using metrics like accuracy, precision, recall, and F1-score.



Types of Machine Learning

1. Supervised Learning – Models learn from labeled data.

Examples:

Classification (e.g., spam detection, image recognition)

Regression (e.g., predicting house prices, stock market trends)




2. Unsupervised Learning – Models learn from unlabeled data to find patterns.

Examples:

Clustering (e.g., customer segmentation, anomaly detection)

Dimensionality Reduction (e.g., PCA, t-SNE)




3. Reinforcement Learning – Models learn through trial and error using rewards.

Examples:

Robotics

Game playing (e.g., AlphaGo)





Common ML Algorithms

Linear Regression – Predicts continuous values

Logistic Regression – Used for binary classification

Decision Trees – Splits data into branches for decision making

Random Forest – An ensemble of decision trees

Support Vector Machines (SVM) – Finds the best boundary between classes

Neural Networks – Mimics the human brain for deep learning


Model Evaluation Metrics

Accuracy – Correct predictions out of total predictions

Precision – How many predicted positives are actually positive

Recall – How many actual positives were correctly predicted

F1 Score – Harmonic mean of precision and recall


Machine Learning Workflow

1. Data Collection


2. Data Preprocessing (Cleaning, Feature Engineering)


3. Model Selection


4. Model Training


5. Model Evaluation


6. Model Deployment



*************************†********************************************





Supervised Machine Learning

Supervised learning is a type of machine learning where a model is trained using labeled data. The model learns from input-output pairs, making predictions based on patterns it finds in the data.

Key Components

1. Input (Features) – Independent variables used for prediction.


2. Output (Target/Label) – The value we want to predict.


3. Training Data – Labeled dataset used to train the model.


4. Testing Data – Separate data used to evaluate model performance.



Types of Supervised Learning

1. Regression – Predicts continuous numerical values.

Example: Predicting house prices based on size, location, etc.

Algorithms:

Linear Regression

Decision Tree Regression

Random Forest Regression



2. Classification – Predicts categorical labels.

Example: Spam detection (spam or not spam), disease prediction (positive or negative).

Algorithms:

Logistic Regression

Decision Trees

Support Vector Machines (SVM)

Random Forest

Neural Networks



Supervised Learning Workflow

1. Data Collection – Gather labeled data.


2. Data Preprocessing – Handle missing values, normalize data, and split into training/testing sets.


3. Model Selection – Choose an appropriate algorithm.


4. Model Training – Train the model using labeled data.


5. Model Evaluation – Assess performance using metrics like accuracy, precision, recall, RMSE (for regression).


6. Prediction – Use the trained model for making predictions.



Example: Supervised Learning in Python (Classification using Logistic Regression)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset (Iris dataset)
from sklearn.datasets import load_iris
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

This example trains a logistic regression model on the Iris dataset and evaluates its accuracy.

Would you like a hands-on example for a specific dataset or problem?



