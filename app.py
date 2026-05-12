import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction System")

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

# Get feature names and importances
feature_names = model.feature_names_in_
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Dashboard"])

with tab1:
    # ===============================
    # UPLOAD CSV
    # ===============================
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(data.head())

        # ===============================
        # STORE TARGET IF EXISTS (FOR ROC CURVE)
        # ===============================
        y_true = None
        if "Churn" in data.columns:
            y_true = data["Churn"].copy()
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

            # Risk categories
            def get_risk_category(prob):
                if prob <= 0.3:
                    return "Low Risk"
                elif prob <= 0.7:
                    return "Medium Risk"
                else:
                    return "High Risk"

            risk_categories = [get_risk_category(p) for p in probs]

            result = data.copy()
            result["Predicted_Churn"] = ["Yes" if p == 1 else "No" for p in preds]
            result["Churn_Probability"] = [f"{p:.0%}" for p in probs]
            result["Risk_Category"] = risk_categories

            st.subheader("📊 Churn Prediction Results")
            st.dataframe(result.head(20))

            # Summary
            churn_count = (preds == 1).sum()
            total = len(preds)
            st.metric("Churn Customers", f"{churn_count} / {total} ({churn_count/total:.1%})")

            # Visualizations
            st.subheader("📈 Visual Insights")

            col1, col2 = st.columns(2)

            with col1:
                # Pie chart for churn distribution
                churn_dist = pd.Series(preds).value_counts()
                fig = px.pie(values=churn_dist.values, names=['Not Churn', 'Churn'] if 0 in churn_dist.index else ['Churn'], title="Churn Distribution")
                st.plotly_chart(fig)

            with col2:
                # Risk categories
                risk_dist = pd.Series(risk_categories).value_counts()
                fig = px.bar(x=risk_dist.index, y=risk_dist.values, title="Risk Categories")
                st.plotly_chart(fig)

            # Feature importance
            st.subheader("🔍 Key Factors Affecting Churn")
            st.bar_chart(feature_importance_df.set_index('Feature')['Importance'])

            # ROC Curve (if ground truth is available)
            if y_true is not None:
                st.subheader("📈 ROC Curve & Model Performance")
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_true, probs)
                roc_auc = auc(fpr, tpr)
                
                # Create ROC curve plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.3f})',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier (AUC = 0.500)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    title='ROC Curve - Model Performance',
                    hovermode='closest'
                )
                st.plotly_chart(fig)
                st.metric("AUC Score", f"{roc_auc:.3f}")

                # Confusion Matrix
                st.subheader("📊 Confusion Matrix")
                cm = confusion_matrix(y_true, preds)
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted: No Churn', 'Predicted: Churn'],
                    y=['Actual: No Churn', 'Actual: Churn'],
                    text=cm,
                    texttemplate='%{text}',
                    colorscale='Blues'
                ))
                fig.update_layout(title='Confusion Matrix - TP, TN, FP, FN')
                st.plotly_chart(fig)

                # Precision-Recall Curve
                st.subheader("📈 Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(y_true, probs)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name='Precision-Recall Curve',
                    line=dict(color='green', width=3)
                ))
                fig.update_layout(
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    title='Precision-Recall Curve',
                    hovermode='closest'
                )
                st.plotly_chart(fig)

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

            # Risk category
            if prob <= 0.3:
                risk = "Low Risk"
            elif prob <= 0.7:
                risk = "Medium Risk"
            else:
                risk = "High Risk"

            # Prediction result
            if pred == 1:
                st.error(f"🚨 Likely to Churn")
            else:
                st.success(f"✅ Not Likely to Churn")

            st.write(f"**Churn Probability:** {prob:.0%}")
            st.write(f"**Risk Level:** {risk}")

            # Important factors (top 3 global)
            st.subheader("🔍 Key Factors Affecting Churn")
            top_factors = feature_importance_df.head(3)['Feature'].tolist()
            st.write("Based on model analysis, the top factors influencing churn are:")
            for factor in top_factors:
                st.write(f"- {factor}")

            # Recommended actions
            st.subheader("💡 Recommended Actions")
            if pred == 1:
                if risk == "High Risk":
                    st.write("- Offer immediate discount (20-30%) on monthly charges")
                    st.write("- Provide premium customer support")
                    st.write("- Schedule a personal call to understand concerns")
                elif risk == "Medium Risk":
                    st.write("- Send personalized retention email with benefits")
                    st.write("- Offer loyalty rewards or upgrades")
                else:
                    st.write("- Monitor usage and send satisfaction surveys")
            else:
                st.write("- Continue providing excellent service")
                st.write("- Encourage referrals or upgrades")

            # Simulate email alert
            if pred == 1 and st.button("📧 Send Retention Email Alert"):
                st.subheader("📧 Simulated Email Alert")
                st.info(f"""
                **To:** customer@example.com  
                **Subject:** We Value You - Special Retention Offer  

                Dear Customer,  

                We noticed some potential concerns with your service. To ensure you continue enjoying our services, we're offering you a special 25% discount on your next month's bill.  

                Additionally, our premium support team is available 24/7 to assist you.  

                Best regards,  
                Customer Retention Team
                """)

with tab2:
    st.header("📊 Business Insights Dashboard")

    # Load training data for insights
    try:
        train_data = pd.read_csv("data/customer_churn_dataset-training-master.csv")
        st.success("Training data loaded for insights")

        # ===============================
        # SECTION 1: CHURN OVERVIEW
        # ===============================
        st.subheader("1️⃣ Churn Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            churn_rate = train_data['Churn'].sum() / len(train_data)
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        
        with col2:
            retention_rate = 1 - churn_rate
            st.metric("Retention Rate", f"{retention_rate:.1%}")
        
        with col3:
            total_customers = len(train_data)
            st.metric("Total Customers", total_customers)

        col1, col2 = st.columns(2)

        with col1:
            # Pie Chart: Churn Distribution
            churn_counts = train_data['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=['Not Churn', 'Churn'],
                title="Churned vs Retained Customers",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Risk Level Distribution (simulated based on churn)
            risk_dist = train_data['Churn'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk').value_counts()
            fig = px.bar(
                x=risk_dist.index,
                y=risk_dist.values,
                title="Risk Level Distribution",
                labels={'x': 'Risk Level', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 2: DEMOGRAPHIC ANALYSIS
        # ===============================
        st.subheader("2️⃣ Demographic Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Churn by Gender
            gender_churn = train_data.groupby(['Gender', 'Churn'], observed=False).size().unstack(fill_value=0)
            fig = px.bar(
                gender_churn,
                title="Churn by Gender",
                barmode='group',
                labels={'value': 'Count', 'index': 'Gender'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn by Age Group
            train_data['Age_Group'] = pd.cut(train_data['Age'], bins=[0, 30, 50, 100], labels=['Young (0-30)', 'Middle-aged (30-50)', 'Senior (50+)'])
            age_churn = train_data.groupby(['Age_Group', 'Churn'], observed=False).size().unstack(fill_value=0)
            fig = px.bar(
                age_churn,
                title="Churn by Age Group",
                barmode='group',
                labels={'value': 'Count', 'index': 'Age Group'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Age Distribution Histogram
        fig = px.histogram(
            train_data,
            x='Age',
            title='Distribution of Customer Age',
            nbins=20,
            labels={'Age': 'Age', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 3: SUBSCRIPTION & PLAN ANALYSIS
        # ===============================
        st.subheader("3️⃣ Subscription & Plan Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Churn by Subscription Type
            subscription_churn = train_data.groupby(['Subscription Type', 'Churn'], observed=False).size().unstack(fill_value=0)
            fig = px.bar(
                subscription_churn,
                title="Churn by Subscription Type",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn by Contract Length
            contract_churn = train_data.groupby(['Contract Length', 'Churn'], observed=False).size().unstack(fill_value=0)
            fig = px.bar(
                contract_churn,
                title="Churn by Contract Length",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 4: FINANCIAL ANALYSIS
        # ===============================
        st.subheader("4️⃣ Financial Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Total Spend vs Churn (Scatter Plot)
            fig = px.scatter(
                train_data,
                x='Total Spend',
                y='Usage Frequency',
                color='Churn',
                title='Total Spend vs Usage Frequency',
                labels={'Total Spend': 'Total Spend ($)', 'Usage Frequency': 'Usage Frequency', 'Churn': 'Churned'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Revenue Loss Chart
            revenue_by_churn = train_data.groupby('Churn')['Total Spend'].sum()
            fig = px.bar(
                x=['Retained', 'Churned'],
                y=revenue_by_churn.values,
                title='Total Revenue by Customer Status',
                labels={'x': 'Customer Status', 'y': 'Revenue ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Distribution of Total Spend
        fig = px.histogram(
            train_data,
            x='Total Spend',
            title='Distribution of Total Spend',
            nbins=20,
            labels={'Total Spend': 'Total Spend ($)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 5: CUSTOMER BEHAVIOR ANALYSIS
        # ===============================
        st.subheader("5️⃣ Customer Behavior Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Tenure vs Churn (Scatter Plot)
            fig = px.scatter(
                train_data,
                x='Tenure',
                y='Usage Frequency',
                color='Churn',
                title='Tenure vs Usage Frequency',
                labels={'Tenure': 'Tenure (months)', 'Usage Frequency': 'Usage Frequency', 'Churn': 'Churned'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Usage Frequency Distribution
            fig = px.histogram(
                train_data,
                x='Usage Frequency',
                title='Distribution of Usage Frequency',
                nbins=20,
                labels={'Usage Frequency': 'Usage Frequency', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Support Calls vs Churn
        support_churn = train_data.groupby(['Support Calls', 'Churn'], observed=False).size().unstack(fill_value=0)
        fig = px.bar(
            support_churn,
            title='Churn by Number of Support Calls',
            barmode='group',
            labels={'Support Calls': 'Support Calls', 'value': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 6: FEATURE ANALYSIS
        # ===============================
        st.subheader("6️⃣ Feature Importance & Correlation")
        
        col1, col2 = st.columns(2)

        with col1:
            # Feature Importance
            fig = px.bar(
                feature_importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Features Affecting Churn'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top Churn Reasons (based on feature importance)
            st.subheader("🔍 Top Churn Drivers")
            top_factors = feature_importance_df.head(5)
            for idx, row in top_factors.iterrows():
                st.write(f"**{idx+1}. {row['Feature']}** (Importance: {row['Importance']:.3f})")

        # Correlation Heatmap
        st.subheader("Correlation Matrix Heatmap")
        numeric_data = train_data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title='Feature Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 7: CUSTOMER SEGMENTATION
        # ===============================
        st.subheader("7️⃣ Customer Segmentation")
        
        # Segment by Total Spend
        train_data['Spending_Segment'] = pd.qcut(train_data['Total Spend'], q=3, labels=['Low Spender', 'Medium Spender', 'High Spender'])
        spending_churn = train_data.groupby(['Spending_Segment', 'Churn'], observed=False).size().unstack(fill_value=0)
        fig = px.bar(
            spending_churn,
            title='Churn by Customer Spending Segment',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 8: OUTLIER DETECTION
        # ===============================
        st.subheader("8️⃣ Outlier Detection - Box Plots")
        col1, col2 = st.columns(2)

        with col1:
            # Box Plot for Payment Delay
            fig = px.box(
                train_data,
                y='Payment Delay',
                title='Distribution of Payment Delay (Outliers)',
                labels={'Payment Delay': 'Payment Delay (days)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box Plot for Total Spend
            fig = px.box(
                train_data,
                y='Total Spend',
                title='Distribution of Total Spend (Outliers)',
                labels={'Total Spend': 'Total Spend ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Box Plot for Usage Frequency
        fig = px.box(
            train_data,
            y='Usage Frequency',
            title='Distribution of Usage Frequency (Outliers)',
            labels={'Usage Frequency': 'Usage Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # SECTION 9: HIGH-RISK CUSTOMERS
        # ===============================
        st.subheader("9️⃣ High-Risk Customer Insights")
        high_risk = train_data[train_data['Churn'] == 1].sort_values('Total Spend', ascending=False)
        st.write(f"**Total Churned Customers:** {len(high_risk)}")
        st.write(f"**Revenue Lost:** ${high_risk['Total Spend'].sum():,.2f}")
        st.dataframe(
            high_risk[['CustomerID', 'Age', 'Gender', 'Tenure', 'Total Spend', 'Payment Delay', 'Support Calls']].head(15).reset_index(drop=True),
            use_container_width=True
        )

    except FileNotFoundError:
        st.error("Training data not found for dashboard")
