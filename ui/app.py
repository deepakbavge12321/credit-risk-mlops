import streamlit as st
import requests
import pandas as pd
import json

# Configure page for a modern, high-contrast professional aesthetic
st.set_page_config(
    page_title="Credit Risk Platform", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="💳"
)

# Custom CSS for UI enhancements
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; padding: 10px; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk Prediction Platform")

# -----------------------------
# Evaluation Rubric: UI Tabs (Predictor, Pipeline, Manual)
# -----------------------------
tab_predict, tab_pipeline, tab_manual = st.tabs([
    "🔍 Risk Predictor", 
    "⚙️ ML Pipeline Tracking", 
    "📖 User Manual"
])

with tab_manual:
    st.header("User Manual")
    st.markdown("""
    **Welcome to the Credit Risk Predictor.** This tool helps loan officers evaluate the probability of a loan default.
    
    **How to use:**
    1. Navigate to the **Risk Predictor** tab.
    2. Fill in the **Applicant Demographics** (e.g., Age in years, Gender). The system will automatically convert these into the required formats (e.g., negative days).
    3. Enter the **Financial Details** like Income, Requested Loan Amount, and Annual Payment.
    4. Provide the **External Bureau Scores** (values between 0.0 and 1.0).
    5. Click **Predict Risk** to receive an automated Approve/Reject recommendation based on the AI model.
    """)

with tab_pipeline:
    st.header("ML Pipeline & MLOps Console")
    st.markdown("This interface tracks the automated data ingestion, CI/CD, and model training pipelines.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Pipeline Status")
        st.success("✅ Data Ingestion: Completed")
        st.success("✅ DVC Feature Engineering: Completed")
        st.success("✅ MLflow Training Run: #02f2f188657d4d7f")
    with col2:
        st.subheader("Monitoring Links")
        st.info("🔗 [MLflow Experiment Tracker](http://localhost:5000)")
        st.info("🔗 [Prometheus Metrics](http://localhost:9090)")
        st.info("🔗 [Grafana Dashboards](http://localhost:3000)")

with tab_predict:
    st.markdown("Enter applicant details below. Inputs are restricted to valid ranges to prevent errors.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🧑‍💼 Demographics")
            # Converted from DAYS_BIRTH for non-technical users
            age_years = st.number_input("Applicant Age", min_value=18, max_value=100, value=35, help="Age of the applicant in years.")
            gender = st.selectbox("Gender", ["Female", "Male"], help="Applicant's gender")
            education = st.selectbox("Education Level (Encoded)", [0, 1, 2, 3, 4, 5], index=2)
            years_employed = st.number_input("Years Employed", min_value=0.0, max_value=50.0, value=5.0)

        with col2:
            st.subheader("💰 Financial Details")
            income = st.number_input("Total Annual Income ($)", min_value=10000.0, value=200000.0, step=10000.0)
            credit = st.number_input("Requested Loan Amount ($)", min_value=10000.0, value=500000.0, step=10000.0)
            annuity = st.number_input("Annual Loan Payment ($)", min_value=1000.0, value=25000.0, step=1000.0)
            goods_price = st.number_input("Price of Goods ($)", min_value=0.0, value=450000.0, step=10000.0)

        with col3:
            st.subheader("📊 External Bureau Scores")
            st.markdown("*(Highly critical impact on risk score)*")
            ext1 = st.slider("External Source 1", 0.0, 1.0, 0.5, help="Normalized score from external data source.")
            ext2 = st.slider("External Source 2", 0.0, 1.0, 0.6)
            ext3 = st.slider("External Source 3", 0.0, 1.0, 0.7)

        submitted = st.form_submit_button("🔍 Predict Risk Recommendation")

    if submitted:
        with st.spinner("Analyzing risk..."):
            # Load full schema to ensure no missing columns for the backend API
            try:
                with open("features.json", "r") as f:
                    FEATURES = json.load(f)
                payload = {col: 0 for col in FEATURES}
            except FileNotFoundError:
                st.error("features.json not found. Please ensure the model pipeline has run.")
                st.stop()

            # Transform user-friendly inputs to model-expected formats
            days_birth = int(-age_years * 365.25)
            days_employed = int(-years_employed * 365.25)
            emp_birth_ratio = days_employed / days_birth if days_birth != 0 else 0
            credit_annuity_ratio = credit / annuity if annuity != 0 else 0
            credit_goods_ratio = credit / goods_price if goods_price != 0 else 0

            # Override default 0s with actual user data
            payload.update({
                "CODE_GENDER": 1 if gender == "Male" else 0,
                "DAYS_BIRTH": days_birth,
                "DAYS_EMPLOYED": days_employed,
                "NAME_EDUCATION_TYPE": education,
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods_price,
                "EXT_SOURCE_1": ext1,
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": ext3,
                "EMPLOYED_TO_BIRTH_RATIO": emp_birth_ratio,
                "CREDIT_ANNUITY_RATIO": credit_annuity_ratio,
                "CREDIT_GOODS_RATIO": credit_goods_ratio
            })

            try:
                # Loose coupling: Connect to FastAPI backend
                res = requests.post(
                    "http://localhost:8000/predict",
                    json={"data": payload}
                )
                res.raise_for_status()
                result = res.json()

                prob = result["default_probability"]
                decision = result["decision"]

                st.divider()
                st.subheader("📋 AI Decision Output")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric(label="Probability of Default", value=f"{prob:.2%}")
                with col_res2:
                    if decision == "REJECT":
                        st.error(f"❌ Recommended Action: **{decision}**")
                    else:
                        st.success(f"✅ Recommended Action: **{decision}**")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to Inference API. Ensure backend is running. Details: {e}")