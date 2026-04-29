import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction")

# ===============================
# LOAD MODEL & ENCODERS
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

model, encoders = load_artifacts()
st.success("✔ Model & encoders loaded")

# ===============================
# UPLOAD CSV
# ===============================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(data.head())

    # ===============================
    # DROP TARGET IF EXISTS
    # ===============================
    if "Churn" in data.columns:
        data.drop(columns=["Churn"], inplace=True)

    # ===============================
    # HANDLE NaN FIRST (CRITICAL)
    # ===============================
    data = data.replace({np.nan: "UNKNOWN"})

    # ===============================
    # SAFE LABEL ENCODING
    # ===============================
    for col, le in encoders.items():
        if col in data.columns:
            # Convert to string (IMPORTANT)
            data[col] = data[col].astype(str)

            # Add UNKNOWN to encoder if missing
            if "UNKNOWN" not in le.classes_:
                le.classes_ = np.append(le.classes_, "UNKNOWN")

            # Replace unseen values
            data[col] = data[col].apply(
                lambda x: x if x in le.classes_ else "UNKNOWN"
            )

            # Transform
            data[col] = le.transform(data[col])

    # ===============================
    # CONVERT REMAINING OBJECT COLS
    # ===============================
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    # ===============================
    # FEATURE ORDER MATCH
    # ===============================
    data = data[model.feature_names_in_]

    # ===============================
    # PREDICT
    # ===============================
    if st.button("🚀 Predict Churn"):
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]

        result = data.copy()
        result["Predicted_Churn"] = preds
        result["Churn_Probability"] = probs.round(2)

        st.subheader("Results")
        st.dataframe(result.head(20))

        st.warning(f"🚨 Churn Customers: {(preds == 1).sum()} / {len(preds)}")

        st.download_button(
            "⬇ Download Results",
            result.to_csv(index=False),
            "churn_predictions.csv",
            "text/csv"
        )

# ===============================
# MANUAL PREDICTION
# ===============================
st.header("🧮 Manual Prediction")

with st.form("manual"):
    user_input = {}
    for col in model.feature_names_in_:
        user_input[col] = st.text_input(col)

    submit = st.form_submit_button("Predict")

    if submit:
        df = pd.DataFrame([user_input])
        df = df.replace({np.nan: "UNKNOWN"})

        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)

                if "UNKNOWN" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "UNKNOWN")

                df[col] = df[col].apply(
                    lambda x: x if x in le.classes_ else "UNKNOWN"
                )
                df[col] = le.transform(df[col])

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = df[model.feature_names_in_]

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        if pred == 1:
            st.error(f"🚨 Likely to Churn (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Not Likely to Churn (Probability: {prob:.2f})")
