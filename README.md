# 📊 Customer Churn Prediction System

## 🚀 Overview
This project predicts whether a customer is likely to churn using a Machine Learning model.  
It provides an interactive web interface built with Streamlit where users can upload data or enter inputs manually to get predictions.

---

## 🧠 Features
- 📂 Upload CSV file for bulk predictions  
- 🧮 Manual input for single prediction  
- 📈 Displays churn probability  
- 📊 Shows prediction results in table format  
- ⬇ Download prediction results as CSV  

---

## 🛠️ Tech Stack
- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  
- Joblib  

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
1. Upload a dataset OR enter customer details manually  
2. Click on **Predict Churn**  
3. View prediction and probability  
4. Download results if needed  

---

## 📊 Output
- **Prediction:** Churn / Not Churn  
- **Probability Score:** Likelihood of churn  

---

## 🔮 Future Improvements
- Convert to full-stack app (React + FastAPI)  
- Deploy on cloud (AWS / Render / Streamlit Cloud)  
- Add authentication system  
- Improve UI/UX  

---

## ⚠️ Note
- Ensure model (`.pkl`) and encoder files are present in the root directory  
- Compatible with specific scikit-learn version used during training  

---

## 👨‍💻 Author
**Prashant Kumar Tripathi**  
B.Tech CSE Student
