# src/train_model.py

"""
Train a Decision Tree model on the Kepler exoplanet dataset and save it for later use.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1️⃣ LOAD THE DATA
# ================================
DATA_PATH = "../data/kepler_data.csv"   # change to your CSV name if needed
df = pd.read_csv(DATA_PATH)

print("✅ Data loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ================================
# 2️⃣ SET TARGET AND FEATURES
# ================================
target_col = "exoplanet_archive_disposition"   # confirm this is the exact column name
FEATURES = [
    'disposition_score', 'ntl_fpflag', 
    'se_fpflag', 'co_fpflag', 'ec_fpflag',
    'koi_period', 'koi_depth',
    'koi_prad', 'koi_eqtemp',
    'koi_stefftemp', 'koi_srad',
    'koi_slogg', 'koi_kepmag'
]

X = df[FEATURES]
y = df[target_col]

# ================================
# 3️⃣ ENCODE TARGET LABELS
# ================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ================================
# 4️⃣ HANDLE NUMERIC FEATURES ONLY
# ================================
# drop non-numeric columns like names, flags, text fields
X = X.select_dtypes(include=["float64", "int64"])

# fill missing numeric values with median (keeps all rows!)
X = X.fillna(X.median())

# ================================
# 5️⃣ SCALE FEATURES
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 6️⃣ TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ================================
# 7️⃣ TRAIN MODEL
# ================================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# ================================
# 8️⃣ EVALUATE MODEL
# ================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {acc:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.tight_layout()
plt.savefig("../models/confusion_matrix2.png")
plt.close()

# Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(15), y=importances.head(15).index)
plt.title("Top 15 Most Important Features")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../models/feature_importances2.png")
plt.close()

print("\n📊 Saved evaluation plots to '../models/' folder.")

# ================================
# 9️⃣ SAVE MODEL + SCALER + LABEL ENCODER
# ================================
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/trained_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(le, "../models/label_encoder.pkl")
joblib.dump(FEATURES, "../models/feature_list.pkl")


print("\n💾 Model, scaler, and label encoder saved successfully!")
print("📁 Files saved in '../models/' folder:")
print(" - trained_model.pkl")
print(" - scaler.pkl")
print(" - label_encoder.pkl")
print(" - feature_list.pkl")
print(" - confusion_matrix.png")
print(" - feature_importances.png")
