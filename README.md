# Sentiment-ML-Model
An ML model developed in python taking advantage of sklearn for training multiple models for a dataset filled out by hundreds of people with dozens of features from sentiment classing, feelings, numerical fields, free text, etc... Validated in multiple ways to determine the best model, and testing the final model with 90% test accuracy.

KNN, Logistic Regression (LR), Bernoulli Naive Bayes (BNN), and Random Forest were trained.
Validation led to a stacked model of LR for numerical and continuous features and BNN for free-text based features from the dataset to be used, culminating in a final model.

sklearn was used for testing, however there are manual implementations of all models with no external ML library used, only training weights from the trained sklearn models were used (not included).
