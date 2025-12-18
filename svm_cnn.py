import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score



#Load MobileNet feature files
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


#Train SVM
svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=False
)

print("Training SVM")
svm.fit(X_train_scaled, y_train)


# Unknown class detection function
def predict_with_unknown(model, X_scaled, threshold):
    scores = model.decision_function(X_scaled)

    confidence = np.max(scores, axis=1)
    predictions = model.predict(X_scaled)

    predictions_with_unknown = np.where(confidence < threshold, 6, predictions)

    return predictions_with_unknown

# Evaluate
y_pred = svm.predict(X_val_scaled)

acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.4f}")
scores = svm.decision_function(X_val_scaled)


# Calculate confidence scores for the validation set
val_scores = np.max(svm.decision_function(X_val_scaled), axis=1)

# Calculate Mean and Standard Deviation
mu = np.mean(val_scores)
std = np.std(val_scores)

# We define the threshold 

buffer = max(3 * std, mu * 0.05)
optimal_threshold = mu - buffer

print(f"--- Statistical Analysis ---")
print(f"Mean Confidence: {mu:.4f}")
print(f"Confidence StdDev: {std:.4f}")
print(f"Mathematically Derived Threshold: {optimal_threshold:.4f}")

# Apply the new threshold
y_pred_final = predict_with_unknown(svm, X_val_scaled, threshold=optimal_threshold)

print(f"\n--- Final Model Performance ---")
print(classification_report(y_val, y_pred_final,labels=[0, 1, 2, 3, 4, 5, 6], target_names=[
    "Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Unknown (6)"
]))

# Check how many were labeled as unknown
unknown_count = np.sum(y_pred_final == 6)
print(f"Samples categorized as 'Unknown': {unknown_count} out of {len(y_val)}")


# Save Model + Scaler
os.makedirs("models", exist_ok=True)
# Create a metadata dictionary
model_metadata = {
    "model": svm,
    "scaler": scaler,
    "threshold": optimal_threshold,
    "feature_type": "MobileNetV2",
    "classes": ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Unknown"]
}

# Save 
joblib.dump(model_metadata, "models/final_svm_package.pkl")
print("\nFinal package with adaptive threshold saved to 'models/final_svm_package.pkl'")


