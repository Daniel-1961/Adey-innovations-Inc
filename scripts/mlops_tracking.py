import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_experiment("Fraud Detection Model")

def train_and_log_model(X_train, X_test, y_train, y_test, model_name):
    """
    Trains a RandomForest model, logs parameters, metrics, and registers the model in MLflow.
    """
    with mlflow.start_run():
        # Define model
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        # Train the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Compute Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        }

        # Log metrics to MLflow
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)

        print("\nModel Training Complete!")
        print("Metrics:", metrics)

    return model
