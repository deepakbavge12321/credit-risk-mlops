import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("💳 Credit Risk Prediction System")

# -----------------------------
# Sidebar Inputs (IMPORTANT FEATURES)
# -----------------------------
st.sidebar.header("Key Risk Features")

# Financial
income = st.sidebar.number_input("AMT_INCOME_TOTAL", value=200000.0)
credit = st.sidebar.number_input("AMT_CREDIT", value=500000.0)
annuity = st.sidebar.number_input("AMT_ANNUITY", value=25000.0)

# External scores (VERY IMPORTANT)
ext1 = st.sidebar.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5)
ext2 = st.sidebar.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6)
ext3 = st.sidebar.slider("EXT_SOURCE_3", 0.0, 1.0, 0.7)

ext_mean = st.sidebar.slider("EXT_SOURCE_MEAN", 0.0, 1.0, 0.6)
ext_min  = st.sidebar.slider("EXT_SOURCE_MIN", 0.0, 1.0, 0.4)
ext_max  = st.sidebar.slider("EXT_SOURCE_MAX", 0.0, 1.0, 0.8)

# Employment / age
days_employed = st.sidebar.number_input("DAYS_EMPLOYED", value=-2000)
days_birth    = st.sidebar.number_input("DAYS_BIRTH", value=-12000)

emp_birth_ratio = st.sidebar.slider("EMPLOYED_TO_BIRTH_RATIO", 0.0, 1.0, 0.2)

# Credit behavior
credit_goods_ratio = st.sidebar.slider("CREDIT_GOODS_RATIO", 0.0, 2.0, 1.0)
credit_annuity_ratio = st.sidebar.slider("CREDIT_ANNUITY_RATIO", 0.0, 50.0, 20.0)

# Bureau features
bureau_debt_ratio = st.sidebar.slider("BUREAU_DEBT_RATIO_MEAN", 0.0, 2.0, 0.5)
bureau_debt_credit_ratio = st.sidebar.slider("BUREAU_DEBT_CREDIT_RATIO", 0.0, 2.0, 0.5)

# Car / demographics
car_age = st.sidebar.number_input("OWN_CAR_AGE", value=5)
car_birth_ratio = st.sidebar.slider("CAR_TO_BIRTH_RATIO", 0.0, 1.0, 0.1)

# Encoded categorical
gender = st.sidebar.selectbox("CODE_GENDER (0=F,1=M)", [0, 1])
education = st.sidebar.slider("NAME_EDUCATION_TYPE (encoded)", 0, 5, 2)

# -----------------------------
# Full feature list
# -----------------------------
with open("features.json", "r") as f:
    FEATURES = json.load(f)

# -----------------------------
# Default payload
# -----------------------------
payload = {col: 0 for col in FEATURES}

# -----------------------------
# Override important features
# -----------------------------
payload.update({
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit,
    "AMT_ANNUITY": annuity,

    "EXT_SOURCE_1": ext1,
    "EXT_SOURCE_2": ext2,
    "EXT_SOURCE_3": ext3,
    "EXT_SOURCE_MEAN": ext_mean,
    "EXT_SOURCE_MIN": ext_min,
    "EXT_SOURCE_MAX": ext_max,

    "DAYS_EMPLOYED": days_employed,
    "DAYS_BIRTH": days_birth,
    "EMPLOYED_TO_BIRTH_RATIO": emp_birth_ratio,

    "CREDIT_GOODS_RATIO": credit_goods_ratio,
    "CREDIT_ANNUITY_RATIO": credit_annuity_ratio,

    "BUREAU_DEBT_RATIO_MEAN": bureau_debt_ratio,
    "BUREAU_DEBT_CREDIT_RATIO": bureau_debt_credit_ratio,

    "OWN_CAR_AGE": car_age,
    "CAR_TO_BIRTH_RATIO": car_birth_ratio,

    "CODE_GENDER": gender,
    "NAME_EDUCATION_TYPE": education,
})

# -----------------------------
# Predict
# -----------------------------
if st.button("🔍 Predict Risk"):

    try:
        res = requests.post(
            "http://127.0.0.1:8000/predict" | "http://api:8000/predict",  # Use 'api' when running in Docker
            json=payload
        )

        result = res.json()

        prob = result["default_probability"]
        decision = result["decision"]

        st.subheader("📊 Prediction Result")

        st.metric("Default Probability", f"{prob:.4f}")

        if decision == "REJECT":
            st.error(f"❌ Decision: {decision}")
        else:
            st.success(f"✅ Decision: {decision}")

    except Exception as e:
        st.error(f"API Error: {e}")

# -----------------------------
# Debug
# -----------------------------
with st.expander("Show Payload"):
    st.write(pd.DataFrame([payload]))