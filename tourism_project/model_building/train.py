
"""
Tourism Package Purchase Prediction - Model Training

Loads the train/test splits produced by the data-preparation job,
trains and tunes multiple classification models - Logistic Regression, Decision Tree, Random Forest,
Bagging Classifier, AdaBoost & Gradient Boosting.XGBoost using GridSearchCV;
tracks experiments with MLflow, evaluates the best model on the held-out test set, and saves the overall best model into tourism_project/deployment/.
"""


import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Paths and MLflow configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = "tourism_project/data"
MODEL_DIR = "tourism_project/deployment"

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wellness-Tourism-1")


# --------------------------------------------------
# 2. Load training and testing datasets
# --------------------------------------------------

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")


# Preserve original feature names and column order.
Xtest = Xtest[Xtrain.columns]

print("Training data shape:", Xtrain.shape)
print("Testing data shape:", Xtest.shape)


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

numeric_features = Xtrain.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)


# --------------------------------------------------
# 4. Define seven candidate models and parameter grids
# --------------------------------------------------

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=42),
        {
            "logisticregression__C": [0.1, 1.0, 10.0]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "decisiontreeclassifier__max_depth": [5, 10, None],
            "decisiontreeclassifier__min_samples_split": [2, 5, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=1),
        {
            "randomforestclassifier__n_estimators": [100, 200],
            "randomforestclassifier__max_depth": [5, 10, None],
            "randomforestclassifier__min_samples_split": [2, 5]
        }
    ),

    "Bagging": (
        BaggingClassifier(random_state=42, n_jobs=1),
        {
            "baggingclassifier__n_estimators": [50, 100],
            "baggingclassifier__max_samples": [0.7, 1.0],
            "baggingclassifier__max_features": [0.7, 1.0]
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "adaboostclassifier__n_estimators": [50, 100, 200],
            "adaboostclassifier__learning_rate": [0.05, 0.5, 1.0]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "gradientboostingclassifier__n_estimators": [100, 200],
            "gradientboostingclassifier__learning_rate": [0.05, 0.1],
            "gradientboostingclassifier__max_depth": [2, 3, 5]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=1
        ),
        {
            "xgbclassifier__n_estimators": [100, 200],
            "xgbclassifier__max_depth": [3, 5, 7],
            "xgbclassifier__learning_rate": [0.05, 0.1]
        }
    )
}


# --------------------------------------------------
# 5. Train, tune and evaluate each model
# --------------------------------------------------

results = []
best_overall_f1 = -1.0
best_overall_model = None
best_overall_name = None

for model_name, (classifier, param_grid) in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    # Create a fresh preprocessing pipeline for each model.
    pipeline = make_pipeline(preprocessor, classifier)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise"
    )

    with mlflow.start_run(run_name=model_name):

        # Hyperparameter tuning and training.
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_

        # Evaluate on the held-out test set.
        y_pred = best_model.predict(Xtest)

        accuracy = accuracy_score(ytest, y_pred)
        precision = precision_score(
            ytest, y_pred, zero_division=0
        )
        recall = recall_score(
            ytest, y_pred, zero_division=0
        )
        f1 = f1_score(
            ytest, y_pred, zero_division=0
        )

        # Log best hyperparameters.
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("cv_scoring", "f1")

        # Log test metrics.
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric(
            "best_cv_f1", grid_search.best_score_
        )

        # Log ROC AUC when the estimator supports probabilities.
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(Xtest)

            if len(best_model.classes_) == 2:
                try:
                    auc = roc_auc_score(
                        ytest, probabilities[:, 1]
                    )
                    mlflow.log_metric("test_roc_auc", auc)
                except ValueError:
                    print("ROC AUC could not be calculated.")

        # Create signature and example for MLflow.
        input_example = Xtrain.head(5)
        example_predictions = best_model.predict(input_example)
        signature = infer_signature(
            input_example, example_predictions
        )

        # Modern MLflow model logging API.
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example
        )

        print("Best parameters:", grid_search.best_params_)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 score:  {f1:.4f}")

        results.append({
            "Model": model_name,
            "CV F1": grid_search.best_score_,
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1": f1
        })

        # Select the overall best model by test F1.
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = best_model
            best_overall_name = model_name


# --------------------------------------------------
# 6. Display comparison and save overall best model
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    by="Test F1", ascending=False
)

print("\n\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")

joblib.dump(best_overall_model, best_model_path)

print("\nBest model:", best_overall_name)
print(f"Best test F1 score: {best_overall_f1:.4f}")
print("Saved best model to:", best_model_path)

# Save comparison results for later review.
results_path = os.path.join(MODEL_DIR, "model_comparison.csv")
results_df.to_csv(results_path, index=False)

print("Saved model comparison to:", results_path)
