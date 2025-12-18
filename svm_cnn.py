import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score



#Load MobileNet feature files

# FEATURE_PATH = "data/cnn_features"
FEATURE_PATH = "data/MobileNet_features"

X_train = joblib.load(os.path.join(FEATURE_PATH, "train_features.pkl"))
y_train = joblib.load(os.path.join(FEATURE_PATH, "train_labels.pkl"))

X_val = joblib.load(os.path.join(FEATURE_PATH, "val_features.pkl"))
y_val = joblib.load(os.path.join(FEATURE_PATH, "val_labels.pkl"))

print("Train features:", X_train.shape)
print("Val features:", X_val.shape)



#Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)


#Train SVM (RBF Kernel)

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=False
)

print("Training SVM")
svm.fit(X_train_scaled, y_train)


# Unknown class detection function
def predict_with_unknown(model, X_scaled, threshold=0.5):
    scores = model.decision_function(X_scaled)
    confidence = np.max(scores, axis=1)
    predictions = model.predict(X_scaled)

    predictions_with_unknown = np.where(confidence < threshold, 6, predictions)

    return predictions_with_unknown




# Evaluate
y_pred = svm.predict(X_val_scaled)

acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.4f}\n")
print(f"minimum confidence scores: {np.min(svm.decision_function(X_val_scaled))}")
print(f"maximum confidence scores: {np.max(svm.decision_function(X_val_scaled))}")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Unknown class detection evaluation
y_pred_unknown = predict_with_unknown(svm, X_val_scaled, 0.5)
acc_unknown = accuracy_score(y_val, y_pred_unknown)
print(f"\nValidation Accuracy with Unknown Detection: {acc_unknown:.4f}\n")
print("Classification Report with Unknown Detection:")
print(classification_report(y_val, y_pred_unknown))


# Save Model + Scaler
os.makedirs("models", exist_ok=True)

# joblib.dump(svm,    "models/svm_cnn_model.pkl")
# joblib.dump(scaler, "models/cnn_scaler.pkl")

joblib.dump(svm,    "models/svm_mobilenet_model.pkl")
joblib.dump(scaler, "models/mobilenet_scaler.pkl")

print("SVM CNN model and scaler saved successfully")
