
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Akhilesh1108/Project", filename="best_predict_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Customer Purchase Prediction App")
st.write("The Customer purchase Prediction App is an internal tool for Visit with Us staff that predicts whether customers are likely to purchase the new package.")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Age (customer's age)", min_value=18, max_value=90)
CityTier = st.number_input("CityTier (Which tier the city is in 1,2,3)", min_value=1, max_value=3)
DurationOfPitch = st.number_input("DurationOfPitch (time taken to complete a pitch)", min_value=1)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting (people visited during the pitch)", min_value=1)
NumberOfFollowups = st.number_input("NumberOfFollowups (total number of follow ups done post pitch)", min_value=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar (prefered property)", min_value=1, max_value=5)
NumberOfTrips = st.number_input("NumberOfTrips (number of trips customer takes anually)", min_value=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore (score indicating the customer's satisfaction)", min_value=1)
MonthlyIncome = st.number_input("MonthlyIncome (gross monthly income of the customer)")
OwnCar = st.selectbox("OwnCar?", ["Yes", "No"])
Passport = st.selectbox("Passport?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting (number of children below 5 yrs)", min_value=0)
TypeofContact = st.selectbox("TypeofContact (How customer contacted)", ["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation (Occupation of customer)", ["Freelancer", "Salaried", "Large Business", "Small Business"])
Gender = st.selectbox("Gender (Gender of customer)", ["Male", "Female"])
MaritalStatus = st.selectbox("MaritalStatus (marital status of cutomer)", ["Married", "Single", "Divorced", "Unmarried"])
Designation = st.selectbox("Designation (work designation)", ["Executive", "Manager", "AVP", "Senior Manager", "VP"])
ProductPitched = st.selectbox("ProductPitched (product pitched)", ["Basic", "Standard", "Deluxe", "King", "Super Delux"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'MonthlyIncome': MonthlyIncome,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Passport': 1 if Passport == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'TypeofContact': {"Company Invited", "Self Enquiry"}[TypeofContact],
    'Occupation': {"Freelancer", "Salaried", "Large Business", "Small Business"}[Occupation],
    'Gender': {"Male", "Female"}[Gender],
    'MaritalStatus': {"Married", "Divorced", "Single", "Unmarried"}[MaritalStatus],
    'Designation': {"Executive", "Manager", "AVP", "Senior Manager", "VP"}[Designation],
    "ProductPitched": {"Basic", "Standard", "Deluxe",  "King", "Super Delux"}[ProductPitched]
}])

if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    result = "Sale Possible" if prediction == 1 else "No Sale"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
