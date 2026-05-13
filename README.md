# Sentiment-ML-Model
An ML model developed in python taking advantage of sklearn for training multiple models for a dataset filled out by hundreds of people with dozens of features from sentiment classing, feelings, numerical fields, free text, etc... Validated in multiple ways to determine the best model, and testing the final model with 90% test accuracy.

KNN, Logistic Regression (LR), Bernoulli Naive Bayes (BNN), and Random Forest were trained.
Validation led to a stacked model of LR for numerical and continuous features and BNN for free-text based features from the dataset to be used, culminating in a final model.

sklearn was used for testing, however there are manual implementations of all models with no external ML library used, only training weights from the trained sklearn models were used (not included).

**Note:** Project done in collaboration with three other developers for a machine learning project. All code created without AI through research and learnt topics. No weights nor datasets included to respect the course's requirements.

Below are relavant investigations and statistics from the models.

<img width="989" height="790" alt="feature_correlation" src="https://github.com/user-attachments/assets/383da914-373c-4f5d-a324-ec0eb3fd8c69" />
<img width="364" height="273" alt="image" src="https://github.com/user-attachments/assets/d1a3e69e-fc47-4a5e-aa87-7467bea5515f" />
<img width="589" height="363" alt="image" src="https://github.com/user-attachments/assets/36eb7aa6-a2a2-4f09-9218-f55c355c4c3b" />
<img width="728" height="359" alt="image" src="https://github.com/user-attachments/assets/d48749ac-89b7-403b-b355-27d8131de2a0" />
<img width="332" height="129" alt="image" src="https://github.com/user-attachments/assets/7fbf7475-0a15-4600-97be-040fe85ff3e8" />
