import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_processed_data():
    """Loads preprocessed datasets for training."""
    datasets = {
        "creditcard": {
            "X_train": pd.read_csv("../data/X_train_cc.csv"),
            "X_test": pd.read_csv("../data/X_test_cc.csv"),
            "y_train": pd.read_csv("../data/y_train_cc.csv"),
            "y_test": pd.read_csv("../data/y_test_cc.csv"),
        },
        "fraud_data": {
            "X_train": pd.read_csv("../data/X_train_fd.csv"),
            "X_test": pd.read_csv("../data/X_test_fd.csv"),
            "y_train": pd.read_csv("../data/y_train_fd.csv"),
            "y_test": pd.read_csv("../data/y_test_fd.csv"),
        }
    }
    return datasets

def train_models(X_train, y_train, dataset_name):
    """Trains multiple models and logs them to MLflow"""
    models = {
        "logistic_regression": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier()
    }

    mlflow.set_experiment("Fraud Detection Experiment")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{dataset_name}_{model_name}"):
            model.fit(X_train, y_train)

            # Save model
            model_path = f"../models/{dataset_name}_{model_name}.pkl"
            joblib.dump(model, model_path)
            
            # Log model
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model_type", model_name)
            mlflow.sklearn.log_model(model, artifact_path=model_name)
            print(f"Saved {model_name} model for {dataset_name}.")

if __name__ == "__main__":
    datasets = load_processed_data()
    
    for dataset_name, data in datasets.items():
        print(f"Training models for: {dataset_name}")
        train_models(data["X_train"], data["y_train"], dataset_name)
