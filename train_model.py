import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("📌 Loading dataset...")
df = pd.read_csv("data/customer_churn_dataset-training-master.csv")
print("✔ Dataset loaded!")

df = df[df["Churn"].notna()]

# ===============================
# ENCODE CATEGORICAL COLUMNS
# ===============================
print("📌 Encoding categorical columns...")

encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# SAVE ENCODERS (IMPORTANT)
joblib.dump(encoders, "label_encoders.pkl")
print("💾 label_encoders.pkl saved!")

# ===============================
# FEATURES & TARGET
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# MODEL TRAINING
# ===============================
print("📌 Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
pred = model.predict(X_test)
print("📊 Accuracy:", accuracy_score(y_test, pred))

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "churn_model.pkl")
print("💾 churn_model.pkl saved!")

# ===============================
# GRAPH
# ===============================
plt.figure()
y.value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.close()

print("✔ churn_distribution.png saved!")

print("\n✅ Training completed successfully!")
