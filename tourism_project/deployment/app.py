"""
tourism_project/deployment/app.py

Streamlit app for "Visit with Us" that loads the model trained and
committed by the pipeline, collects customer details through a simple
form, assembles them into a single-row dataframe and shows whether the
customer is likely to purchase the Wellness Tourism Package.
"""
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧭")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter the customer's details below to predict whether they are likely "
    "to purchase the new Wellness Tourism Package."
)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=2
    )
    NumberOfFollowups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=3
    )
    ProductPitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    DurationOfPitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=0, max_value=180, value=15
    )
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

with col2:
    MaritalStatus = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
    )
    NumberOfTrips = st.number_input(
        "Average Number of Trips per Year", min_value=0, max_value=25, value=3
    )
    Passport = st.selectbox("Holds a Passport?", ["Yes", "No"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0
    )
    Designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    MonthlyIncome = st.number_input(
        "Monthly Income", min_value=0, max_value=200000, value=20000, step=500
    )

if st.button("Predict"):
    input_df = pd.DataFrame(
        [
            {
                "Age": Age,
                "TypeofContact": TypeofContact,
                "CityTier": CityTier,
                "DurationOfPitch": DurationOfPitch,
                "Occupation": Occupation,
                "Gender": Gender,
                "NumberOfPersonVisiting": NumberOfPersonVisiting,
                "NumberOfFollowups": NumberOfFollowups,
                "ProductPitched": ProductPitched,
                "PreferredPropertyStar": PreferredPropertyStar,
                "MaritalStatus": MaritalStatus,
                "NumberOfTrips": NumberOfTrips,
                "Passport": 1 if Passport == "Yes" else 0,
                "PitchSatisfactionScore": PitchSatisfactionScore,
                "OwnCar": 1 if OwnCar == "Yes" else 0,
                "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
                "Designation": Designation,
                "MonthlyIncome": MonthlyIncome,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Tourism Package (probability: {probability:.1%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Tourism Package (probability: {probability:.1%})")
