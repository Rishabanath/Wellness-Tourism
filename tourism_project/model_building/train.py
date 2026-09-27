"""
tourism_project/model_building/train.py

Loads the train/test splits produced by the data-prep job, tunes an XGBoost
classifier with GridSearchCV, tracks every parameter combination and the
final metrics with MLflow, evaluates the best model on the held-out test
set, and saves it into tourism_project/deployment/ so the workflow can
commit it to the repository for the Streamlit app to load.
"""
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")


def load_data():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = make_column_transformer(
        (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            numeric_cols,
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical_cols,
        ),
    )

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    )

    return make_pipeline(preprocessor, model)


def train():
    Xtrain, Xtest, ytrain, ytest = load_data()
    pipeline = build_pipeline(Xtrain)

    # Hyperparameter grid for the XGBoost step of the pipeline
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    mlflow.set_experiment("tourism_wellness_package_prediction")

    with mlflow.start_run(run_name="grid_search_xgboost"):
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(Xtrain, ytrain)

        # Log every tuned parameter combination as its own nested run
        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        y_pred = best_model.predict(Xtest)
        y_proba = best_model.predict_proba(Xtest)[:, 1]

        metrics = {
            "accuracy": accuracy_score(ytest, y_pred),
            "precision": precision_score(ytest, y_pred),
            "recall": recall_score(ytest, y_pred),
            "f1_score": f1_score(ytest, y_pred),
            "roc_auc": roc_auc_score(ytest, y_proba),
        }
        mlflow.log_metrics(metrics)

        print("Best parameters found by GridSearchCV:")
        print(grid_search.best_params_)

        print("\nTest set performance of the best model:")
        for name, value in metrics.items():
            print(f"  {name:>10}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(ytest, y_pred))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.sklearn.log_model(best_model, "model")
        print(f"\nBest model saved to: {MODEL_PATH}")

    return best_model, metrics


if __name__ == "__main__":
    train()
