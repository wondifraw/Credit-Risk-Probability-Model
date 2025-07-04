import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from feature_engineering import process_data_pipeline
from rfm_target_engineering import calculate_rfm, cluster_rfm, assign_high_risk_label, merge_high_risk
import pickle
from mlflow.models import infer_signature


def load_and_prepare_data(raw_data_path=None, filename='clean_data.csv'):
    """
    Loads and processes data, merges RFM high-risk label, and returns features and target.
    Always resolves data path relative to project root.
    """
    try:
        if raw_data_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_data_path = os.path.join(project_root, 'data', 'processed')
        df = process_data_pipeline(raw_data_path=raw_data_path, filename=filename)
        if df is None:
            raise ValueError("Processed DataFrame is None.")
        # Calculate RFM and assign high-risk label
        rfm = calculate_rfm(df)
        rfm = cluster_rfm(rfm)
        rfm = assign_high_risk_label(rfm)
        df = merge_high_risk(df, rfm)
        # Drop columns not needed for modeling
        X = df.drop(columns=['is_high_risk', 'CustomerId', 'TransactionStartTime'], errors='ignore')
        y = df['is_high_risk']
        return X, y
    except Exception as e:
        print(f"Error in load_and_prepare_data: {e}")
        return None, None


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into train and test sets.
    """
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except Exception as e:
        print(f"Error in split_data: {e}")
        return None, None, None, None


def get_models():
    """
    Returns a dictionary of model name to (estimator, param_grid) tuples.
    """
    return {
        'LogisticRegression': (
            LogisticRegression(solver='liblinear', random_state=42),
            {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}
        )
    }


def evaluate_model(y_true, y_pred, y_proba):
    """
    Compute evaluation metrics for classification.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }


def train_and_track(X_train, X_test, y_train, y_test, experiment_name='CreditRiskModel'): 
    """
    Train models, tune hyperparameters, evaluate, and track with MLflow.
    """
    mlflow.set_experiment(experiment_name)
    best_score = -np.inf
    best_model = None
    best_model_name = None
    best_metrics = None
    best_run_id = None
    models = get_models()
    for model_name, (estimator, param_grid) in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            try:
                print(f"Training {model_name}...")
                search = GridSearchCV(estimator, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
                search.fit(X_train, y_train)
                best_estimator = search.best_estimator_
                y_pred = best_estimator.predict(X_test)
                y_proba = best_estimator.predict_proba(X_test)[:, 1]
                metrics = evaluate_model(y_test, y_pred, y_proba)
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics(metrics)
                # Log model with input_example and signature
                input_example = X_train.iloc[:5] if hasattr(X_train, 'iloc') else None
                signature = infer_signature(X_train, y_train) if input_example is not None else None
                mlflow.sklearn.log_model(
                    best_estimator,
                    artifact_path="model",  # Use 'artifact_path' for compatibility; change to 'name' if using latest MLflow
                    input_example=input_example,
                    signature=signature
                )
                print(f"{model_name} metrics: {metrics}")
                if metrics['roc_auc'] > best_score:
                    best_score = metrics['roc_auc']
                    best_model = best_estimator
                    best_model_name = model_name
                    best_metrics = metrics
                    best_run_id = run.info.run_id
            except Exception as e:
                print(f"Error training {model_name}: {e}")
    # Register best model
    if best_model is not None:
        print(f"Registering best model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, f"{experiment_name}_BestModel")
        # --- Save best model to Best_model folder using pickle ---
        os.makedirs('Best_model', exist_ok=True)
        with open(os.path.join('Best_model', f'{best_model_name}.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Best model saved to Best_model/{best_model_name}.pkl")
    else:
        print("No model was successfully trained.")
    return best_model_name, best_metrics


def main():
    X, y = load_and_prepare_data()
    if X is None or y is None:
        print("Data loading failed. Exiting.")
        return
    X_train, X_test, y_train, y_test = split_data(X, y)
    if X_train is None:
        print("Data splitting failed. Exiting.")
        return
    best_model_name, best_metrics = train_and_track(X_train, X_test, y_train, y_test)
    print(f"Best model: {best_model_name}")
    print(f"Best metrics: {best_metrics}")

if __name__ == "__main__":
    main()
