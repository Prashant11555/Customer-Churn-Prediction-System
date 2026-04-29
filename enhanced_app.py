# ===============================
# ENHANCED STREAMLIT APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

from config import FILE_PATHS, VIZ_CONFIG, PREDICTION_CONFIG
from utils import validate_data, encode_categorical_features

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🎯 Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM STYLING
# ===============================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & ENCODERS
# ===============================
@st.cache_resource
def load_artifacts():
    """Load model and encoders"""
    try:
        model = joblib.load(FILE_PATHS["model"])
        encoders = joblib.load(FILE_PATHS["encoders"])
        return model, encoders, "✔ Model loaded successfully"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

model, encoders, status = load_artifacts()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select Page", [
    "🏠 Home",
    "📊 Dashboard",
    "🔮 Predictions",
    "📈 Analytics",
    "ℹ️ About"
])

# ===============================
# HOME PAGE
# ===============================
if page == "🏠 Home":
    st.title("🎯 Customer Churn Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📌 Overview")
        st.info("""
        This system predicts which customers are likely to churn.
        
        **Features:**
        - 🤖 Machine Learning model trained on historical data
        - 📊 Real-time predictions
        - 📈 Comprehensive analytics
        - 🔍 Detailed insights
        """)
    
    with col2:
        st.markdown("### ✅ Status")
        if model is not None:
            st.success(status)
            st.metric("Model Status", "Ready", "✔")
        else:
            st.error(status)
            st.metric("Model Status", "Failed", "❌")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. Go to **Predictions** page to upload CSV and make predictions
    2. Check **Dashboard** for model performance metrics
    3. Visit **Analytics** for detailed insights
    """)

# ===============================
# DASHBOARD PAGE
# ===============================
elif page == "📊 Dashboard":
    st.title("📊 Model Performance Dashboard")
    
    if model is None:
        st.error("❌ Model not loaded. Please train the model first.")
    else:
        # Model Info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "RandomForest", "✔")
        with col2:
            st.metric("Feature Count", len(model.feature_names_in_), "✔")
        with col3:
            st.metric("Status", "Active", "🟢")
        
        st.markdown("---")
        
        # Display visualizations if they exist
        viz_files = {
            "Confusion Matrix": "confusion_matrix.png",
            "ROC Curve": "roc_curve.png",
            "Precision-Recall Curve": "precision_recall_curve.png",
            "Feature Importance": "feature_importance.png"
        }
        
        cols = st.columns(2)
        col_idx = 0
        
        for title, filename in viz_files.items():
            if os.path.exists(filename):
                with cols[col_idx % 2]:
                    st.markdown(f"### {title}")
                    st.image(filename, use_container_width=True)
                col_idx += 1
        
        if col_idx == 0:
            st.warning("⚠️ No visualization files found. Run train_model.py to generate them.")

# ===============================
# PREDICTIONS PAGE
# ===============================
elif page == "🔮 Predictions":
    st.title("🔮 Make Predictions")
    
    if model is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
    else:
        # Upload CSV
        st.markdown("### 📤 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file:
            # Read data
            data = pd.read_csv(uploaded_file)
            st.success(f"✔ File uploaded! Shape: {data.shape}")
            
            # Preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            # Data validation
            st.markdown("### 🔍 Data Validation")
            issues = validate_data(data)
            if issues:
                for issue in issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✔ Data validation passed!")
            
            # Remove target if exists
            if "Churn" in data.columns:
                data = data.drop(columns=["Churn"])
            
            # Encoding
            data_encoded, _ = encode_categorical_features(
                data, 
                encoders=encoders, 
                fit=False
            )
            
            # Handle missing values
            data_encoded = data_encoded.replace({np.nan: 0})
            for col in data_encoded.columns:
                if data_encoded[col].dtype == 'object':
                    data_encoded[col] = pd.to_numeric(
                        data_encoded[col], 
                        errors='coerce'
                    ).fillna(0)
            
            # Feature alignment
            data_encoded = data_encoded[model.feature_names_in_]
            
            # Make predictions
            if st.button("🚀 Generate Predictions", key="predict_btn"):
                with st.spinner("🔄 Making predictions..."):
                    preds = model.predict(data_encoded)
                    probs = model.predict_proba(data_encoded)[:, 1]
                    
                    # Classify risk level
                    risk_levels = []
                    for prob in probs:
                        if prob < 0.3:
                            risk_levels.append("🟢 Low")
                        elif prob < 0.7:
                            risk_levels.append("🟡 Medium")
                        else:
                            risk_levels.append("🔴 High")
                    
                    # Display results
                    result = data.copy()
                    result["Predicted_Churn"] = preds
                    result["Churn_Probability"] = probs.round(4)
                    result["Risk_Level"] = risk_levels
                    
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(result, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(result), "")
                    with col2:
                        st.metric("Churn Predicted", (preds == 1).sum(), 
                                 f"{(preds == 1).sum() / len(preds) * 100:.1f}%")
                    with col3:
                        st.metric("High Risk", (np.array(risk_levels) == "🔴 High").sum(), 
                                 f"{(np.array(risk_levels) == '🔴 High').sum() / len(risk_levels) * 100:.1f}%")
                    with col4:
                        st.metric("Avg Churn Probability", probs.mean().round(2), "")
                    
                    # Download results
                    st.markdown("---")
                    csv = result.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results (CSV)",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
        
        # Manual Prediction
        st.markdown("---")
        st.markdown("### 🧮 Single Customer Prediction")
        
        if st.checkbox("Enter manual values"):
            # Create input fields based on model features
            input_dict = {}
            cols = st.columns(3)
            
            for idx, feature in enumerate(model.feature_names_in_)):
                with cols[idx % 3]:
                    input_dict[feature] = st.number_input(
                        f"Enter {feature}",
                        value=0.0,
                        key=f"feature_{feature}"
                    )
            
            if st.button("🔮 Predict for Customer"):
                input_df = pd.DataFrame([input_dict])
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0, 1]
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Prediction", "Yes ⚠️" if pred == 1 else "No ✔️", "")
                with col2:
                    st.metric("Confidence", f"{prob:.2%}", "")

# ===============================
# ANALYTICS PAGE
# ===============================
elif page == "📈 Analytics":
    st.title("📈 Advanced Analytics")
    
    if model is None:
        st.error("❌ Model not loaded.")
    else:
        st.markdown("### 📊 Feature Importance")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = model.feature_names_in_
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], 
                   color='#2ecc71')
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Table
            st.markdown("### 📋 Feature Importance Scores")
            st.dataframe(importance_df, use_container_width=True)
        else:
            st.warning("⚠️ Model does not have feature importance.")
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Model Statistics")
        st.info(f"""
        **Model Configuration:**
        - Algorithm: RandomForestClassifier
        - Number of Trees: 200
        - Max Depth: 20
        - Features: {len(model.feature_names_in_)}
        - Classes: 2 (No Churn, Churn)
        """)

# ===============================
# ABOUT PAGE
# ===============================
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    
    st.markdown("""
    ### 🎯 Overview
    This Customer Churn Prediction System uses machine learning to identify 
    customers likely to cancel their subscription.
    
    ### 🛠️ Technology Stack
    - **Framework**: Streamlit
    - **ML Library**: Scikit-Learn
    - **Algorithm**: Random Forest Classifier
    - **Languages**: Python 3.8+
    
    ### 📊 Data
    - **Training Data**: Customer churn dataset
    - **Features**: 20+ customer attributes
    - **Target**: Churn (Binary classification)
    
    ### 🚀 Features
    ✅ Real-time predictions
    ✅ Batch processing
    ✅ Model performance metrics
    ✅ Feature importance analysis
    ✅ Risk assessment
    ✅ Data validation
    
    ### 📝 How to Use
    1. Upload a CSV with customer data
    2. System automatically preprocesses and encodes data
    3. Model generates churn predictions
    4. Download results with risk levels
    
    ### 💡 Tips
    - Ensure input CSV has same columns as training data
    - Categorical values will be automatically encoded
    - Probability > 0.7 indicates high churn risk
    
    ---
    **Version**: 2.0 | **Last Updated**: 2026
    """)
