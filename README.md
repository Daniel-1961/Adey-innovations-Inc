
Fraud Detection - End-to-End Machine Learning Project
📌 Overview
This project aims to build a fraud detection system for e-commerce and banking transactions.
It includes data preprocessing, model training, explainability, API development, and a dashboard.

📂 Datasets
Fraud_Data.csv - E-commerce transaction data with fraud labels.
IpAddress_to_Country.csv - Maps IP addresses to country locations.
creditcard.csv - Credit card transaction data (with PCA-transformed features).
📌 Task 1: Data Analysis & Preprocessing
✅ Steps Performed
Handled missing values using imputation techniques.
Removed duplicates and corrected data types.
Performed Exploratory Data Analysis (EDA) to understand fraud trends.
Merged datasets to include geolocation data from IP addresses.
Engineered new features such as:
Transaction frequency
Time-based features (hour of day, day of week)
Normalized numerical features and encoded categorical variables.
📌 Task 2: Model Training
✅ Steps Performed
Split data into training (80%) and testing (20%) sets.
Trained multiple models:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Neural Networks (MLP)
Evaluated models using:
Accuracy, Precision, Recall, F1-score, and AUC-ROC
Used MLflow for experiment tracking and model versioning.
📌 Task 3: Model Explainability
✅ Steps Performed
Used SHAP (Shapley Additive Explanations) to:
Identify important features contributing to fraud detection.
Generate SHAP Summary, Force, and Dependence plots.
Used LIME (Local Interpretable Model-agnostic Explanations) to:
Explain individual fraud predictions.
Generate Feature Importance plots for better interpretability.
📌 Task 4: Model Deployment & API Development
✅ Steps Performed
Built a Flask API to serve fraud predictions.
API supports real-time predictions using a trained model.
Created a Dockerfile to containerize the API.
Steps to build and run the Docker container:
docker build -t fraud-detection-api .
docker run -p 5000:5000 fraud-detection-api

📌 Task 5: Dashboard Devlopment
This task focuses on building an interactive dashboard to visualize fraud detection insights using Dash (Plotly).
The dashboard provides real-time analytics on fraudulent transactions, trends, and distribution.

🎯 Features
✅ Display total transactions, fraud cases, and fraud percentage.
✅ Visualize fraud trends over time using a line chart.
✅ Analyze fraud occurrences across devices and browsers using bar charts.
