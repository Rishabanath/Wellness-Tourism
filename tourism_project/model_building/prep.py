"""
tourism_project/model_building/prep.py

Loads the raw dataset from the repository's data folder, cleans it, drops
columns that carry no predictive signal, and splits it into train/test sets
that are saved locally (as csv files at the repo root) so the workflow can
hand them to the next job as an artifact.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Identifier column: unique per row, carries no predictive signal
DROP_COLUMNS = ["CustomerID"]


def prepare_data(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(path)

    # Drop a stray index column that sometimes gets saved along with the csv
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # Remove unnecessary / identifier columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Basic cleaning: fix an inconsistent category label
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Drop rows where the target itself is missing (can't train/evaluate on them)
    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain: {Xtrain.shape}   Xtest: {Xtest.shape}")
    print(f"ytrain: {ytrain.shape}   ytest: {ytest.shape}")
    print(f"\nTraining target balance:\n{ytrain.value_counts(normalize=True).round(3)}")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()
