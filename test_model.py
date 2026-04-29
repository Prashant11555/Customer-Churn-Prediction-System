import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ===============================
# LOAD TRAINED MODEL
# ===============================
print("📌 Loading trained model...")
model = joblib.load("churn_model.pkl")
print("✔ Model loaded!")

# ===============================
# LOAD TEST DATASET
# ===============================
print("📌 Loading testing dataset...")
test_df = pd.read_csv("data/customer_churn_dataset-testing-master.csv")
print("✔ Testing dataset loaded!")

# ===============================
# HANDLE MISSING VALUES
# ===============================
if "Churn" in test_df.columns:
    test_df = test_df[test_df["Churn"].notna()]

# ===============================
# LOAD SAVED ENCODERS
# ===============================
print("📌 Loading saved encoders...")
encoders = joblib.load("label_encoders.pkl")
print("✔ Encoders loaded!")

# ===============================
# ENCODE CATEGORICAL COLUMNS
# ===============================
print("📌 Encoding categorical columns...")

for col in test_df.select_dtypes(include=["object"]).columns:
    if col in encoders:
        le = encoders[col]
        test_df[col] = le.transform(test_df[col])

print("✔ Encoding complete!")

# ===============================
# FEATURES
# ===============================
if "Churn" in test_df.columns:
    X_test = test_df.drop("Churn", axis=1)
else:
    X_test = test_df.copy()

# ===============================
# PREDICTION
# ===============================
print("📌 Predicting churn...")
predictions = model.predict(X_test)
test_df["Predicted_Churn"] = predictions
print("✔ Prediction completed!")

# ===============================
# SAVE PREDICTIONS
# ===============================
test_df.to_csv("test_predictions.csv", index=False)
print("💾 Test predictions saved to test_predictions.csv")

# ===============================
# GRAPH: CHURN PREDICTION DISTRIBUTION
# ===============================
print("📊 Saving churn prediction graph...")

churn_counts = test_df["Predicted_Churn"].value_counts()

plt.figure()
plt.bar(churn_counts.index.astype(str), churn_counts.values)
plt.xlabel("Predicted Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.title("Test Data – Churn Prediction Distribution")
plt.tight_layout()
plt.savefig("test_churn_distribution.png")
plt.close()

print("✔ Graph saved as test_churn_distribution.png")

# ===============================
# DISPLAY SAMPLE OUTPUT
# ===============================
print("\n================ TEST PREDICTIONS (Sample) ================\n")
print(test_df.head(10).to_string())
print("\n===========================================================")

print("\n✅ Testing completed successfully!")
