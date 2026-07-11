from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Build the full path to the model
MODEL_PATH = BASE_DIR / "customer_churn_pipeline.joblib"

# Load Pipeline
pipeline = joblib.load(MODEL_PATH)


# ---------------- Sidebar ----------------

st.sidebar.title("📊 Customer Churn Prediction")

st.sidebar.markdown("---")

st.sidebar.write("### Model")
st.sidebar.write("Random Forest")

st.sidebar.write("### Accuracy")
st.sidebar.write("82%")

st.sidebar.write("### Developed By")
st.sidebar.write("Arati Mandal")

st.sidebar.markdown("---")

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")

st.markdown(
    "Predict whether a telecom customer is likely to churn based on customer details and subscription information."
)

st.markdown("---")

st.write("Enter customer information below.")

with st.expander("👤 Customer Information", expanded=True):

# -----------------------
# Customer Information
# -----------------------

 col1, col2 = st.columns(2)

 with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

 with col2:
    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

 with col1:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

 with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

 with col1:
    tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

 with col2:
    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

 with col1:
   internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

 with col2:
   security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)



 with col1:
   tech = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)


 with col2:
   contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

# -----------------------
# Prediction
# -----------------------
col1, col2, col3 = st.columns([2,2,2])

with col2:
    predict = st.button(
        "🔍 Predict Churn",
        use_container_width=True
    )

if predict:

    input_df = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone],
        "InternetService":[internet],
        "OnlineSecurity":[security],
        "TechSupport":[tech],
        "Contract":[contract],
        "PaymentMethod":[payment],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total]
    })

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("⚠ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"Probability of Churn: {probability:.2%}")

    if probability < 0.30:
      st.success("🟢 Low Risk Customer")

    elif probability < 0.70:
      st.warning("🟡 Medium Risk Customer")

    else:
      st.error("🔴 High Risk Customer")

      st.subheader("Business Recommendation")

    if prediction == 1:
     st.warning("""
    • Contact the customer

    • Offer a loyalty discount

    • Assign a relationship manager

    • Provide special offers
    """)

    else:
     st.success("""
    • Customer is satisfied

    • Continue regular engagement

    • Recommend premium plans
    """)
     
     st.markdown("---")

    st.info("""
Model Used : Random Forest

Encoding : OneHotEncoder

Deployment : Streamlit

Language : Python

Library : Scikit-Learn
""")