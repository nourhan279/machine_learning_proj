import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ===========================================================
# 1️⃣ Load CNN feature files
# ===========================================================

FEATURE_PATH = "data/cnn_features"

X_train = joblib.load(os.path.join(FEATURE_PATH, "train_features.pkl"))
y_train = joblib.load(os.path.join(FEATURE_PATH, "train_labels.pkl"))

X_val = joblib.load(os.path.join(FEATURE_PATH, "val_features.pkl"))
y_val = joblib.load(os.path.join(FEATURE_PATH, "val_labels.pkl"))

print("Train features:", X_train.shape)
print("Val features:", X_val.shape)


# ===========================================================
# 2️⃣ Feature Scaling (MANDATORY for SVM)
# ===========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)


# ===========================================================
# 3️⃣ Train SVM (RBF Kernel)
# ===========================================================

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=False
)

print("Training SVM...")
svm.fit(X_train_scaled, y_train)


# ===========================================================
# 4️⃣ Evaluate
# ===========================================================

y_pred = svm.predict(X_val_scaled)

acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_val, y_pred))


# ===========================================================
# 5️⃣ Save Model + Scaler
# ===========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(svm,    "models/svm_cnn_model.pkl")
joblib.dump(scaler, "models/cnn_scaler.pkl")

print("SVM CNN model and scaler saved successfully!")
