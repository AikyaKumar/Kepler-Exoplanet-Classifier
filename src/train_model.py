# src/train_model.py
"""
Train the first version (v1) of the Kepler Exoplanet Classifier.
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
import json

# ================================
# 1️⃣ LOAD DATA
# ================================
DATA_PATH = "../data/kepler_data.csv"
df = pd.read_csv(DATA_PATH)

print("✅ Data loaded successfully!")
print("Shape:", df.shape)

# ================================
# 2️⃣ SET TARGET AND FEATURES
# ================================
target_col = "exoplanet_archive_disposition"
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
# 4️⃣ CLEAN + SCALE FEATURES
# ================================
X = X.select_dtypes(include=["float64", "int64"]).fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 5️⃣ SPLIT DATA
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ================================
# 6️⃣ TRAIN MODEL
# ================================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# ================================
# 7️⃣ EVALUATE MODEL
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
plt.savefig("../models/confusion_matrix_v1.png")
plt.close()

# Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(15), y=importances.head(15).index)
plt.title("Top 15 Most Important Features")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../models/feature_importances_v1.png")
plt.close()

# ================================
# 8️⃣ SAVE MODEL + METADATA
# ================================
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/trained_model_v1.pkl")
joblib.dump(scaler, "../models/scaler_v1.pkl")
joblib.dump(le, "../models/label_encoder_v1.pkl")
joblib.dump(FEATURES, "../models/feature_list.pkl")

metadata = {
    "current_version": 1,
    "versions": {
        "1": {
            "accuracy": float(acc),
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
}
with open("../models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n💾 Model v1 saved successfully!")
print("📁 Files saved in '../models/' folder:")
