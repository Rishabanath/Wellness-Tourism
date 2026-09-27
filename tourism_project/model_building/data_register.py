"""
tourism_project/model_building/data_register.py

Registers the raw tourism dataset: verifies that all the columns the rest of
the pipeline depends on are present, and prints a short summary so that the
GitHub Actions log gives a quick sanity check of what was loaded.
"""
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop a stray index column that sometimes gets saved along with the csv
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    print("Dataset registered successfully.")
    print(f"Path : {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("Column dtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).round(3))

    return df


if __name__ == "__main__":
    register_dataset()
