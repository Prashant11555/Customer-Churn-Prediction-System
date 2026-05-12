<<<<<<< HEAD
# 📊 Customer Churn Prediction System

## 🚀 Overview
This project predicts whether a customer is likely to churn using a Machine Learning model.
It provides an interactive web interface built with Streamlit where users can upload data or enter inputs manually to get predictions.
Enhanced with business insights, risk categorization, visualizations, and retention recommendations.

---

## 🧠 Features
- 📂 Upload CSV file for bulk predictions
- 🧮 Manual input for single prediction
- 📈 Displays churn probability and risk category (Low/Medium/High)
- 🔍 Shows key factors affecting churn
- 💡 Provides retention recommendations
- 📊 Interactive dashboard with visualizations (pie charts, bar graphs)
- 📈 Business insights and analytics
- ⬇ Download prediction results as CSV

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Matplotlib
- Seaborn
- Plotly

---

## 📁 Project Structure
ML PROJECT/
├── app.py
├── enhanced_app.py
├── train_model.py
├── enhanced_train_model.py
├── model_evaluation.py
├── utils.py
├── config.py
├── churn_model.pkl
├── label_encoders.pkl
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository

git clone https://github.com/your-username/Customer-Churn-Prediction-System.git

cd Customer-Churn-Prediction-System


### 2️⃣ Create virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate # Windows


### 3️⃣ Install dependencies

pip install -r requirements.txt


### 4️⃣ Run the application

streamlit run app.py


---

## 📌 Usage
1. **Prediction Tab:**
   - Upload a dataset OR enter customer details manually
   - Click on **Predict Churn**
   - View prediction, probability, risk level, key factors, and recommendations
   - Download results if needed

2. **Dashboard Tab:**
   - View business insights and visualizations
   - Analyze churn distribution, risk categories, and key drivers

---

## 📊 Output
- **Prediction:** Likely to Churn / Not Likely to Churn
- **Probability Score:** Likelihood percentage (e.g., 82%)
- **Risk Category:** Low Risk (0-30%), Medium Risk (31-70%), High Risk (71-100%)
- **Key Factors:** Top features influencing churn
- **Recommendations:** Actionable retention strategies
- **Visualizations:** Pie charts, bar graphs, feature importance plots

---

## 🔮 Future Improvements
- Convert to full-stack app (React + FastAPI)
- Deploy on cloud (AWS / Render / Streamlit Cloud)
- Add authentication system
- Improve UI/UX
- Add email/SMS alert generation
- PDF report download
- Real-time prediction API
- Admin analytics panel

---

## ⚠️ Note
- Ensure model (`.pkl`) and encoder files are present in the root directory
- Compatible with specific scikit-learn version used during training  

---

## 👨‍💻 Author
**Prashant Kumar Tripathi**  
B.Tech CSE Student
=======
# 📊 Customer Churn Prediction (Streamlit)

## 🚀 Overview
This project predicts whether a customer is likely to churn using a Machine Learning model.  
It provides an interactive web interface built with Streamlit.

---

## 🧠 Features
- Upload CSV file for batch prediction
- Manual input prediction
- Displays churn probability
- Download prediction results

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

## 📁 Project Structure
>>>>>>> f394d96 (Cleaned requirements for deployment)
