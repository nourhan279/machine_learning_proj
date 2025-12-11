import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ============================
# 1. LOAD HOG FEATURES
# ============================

X_train = joblib.load("data/hog/hog_train_features.pkl")
y_train = joblib.load("data/hog/hog_train_labels.pkl")

X_val = joblib.load("data/hog/hog_val_features.pkl")
y_val = joblib.load("data/hog/hog_val_labels.pkl")

print("Train features:", X_train.shape)
print("Val features:", X_val.shape)


# ============================
# 2. FEATURE SCALING
# ============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Scaling complete.")


# ============================
# 3. TRAIN SVM (RBF KERNEL)
# ============================

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=False,   # we use decision_function NOT probabilities
    decision_function_shape='ovr'
)

svm.fit(X_train_scaled, y_train)

print("SVM training complete!")


# ============================
# 4. NORMAL PREDICTION (NO UNKNOWN YET)
# ============================

val_pred = svm.predict(X_val_scaled)
acc = accuracy_score(y_val, val_pred)

print("\nValidation Accuracy (no Unknown):", acc)
print(classification_report(y_val, val_pred))


# ============================
# 5. UNKNOWN CLASS DETECTION
# ============================
# Project requirement: Any input with low confidence → class 6

def predict_with_unknown(model, scaler, X, threshold):
    X_scaled = scaler.transform(X)
    
    # get decision function scores
    scores = model.decision_function(X_scaled)

    preds = []
    for i in range(len(scores)):
        sample_scores = scores[i]
        max_score = np.max(sample_scores)

        if max_score < threshold:
            preds.append(6)    # Unknown
        else:
            preds.append(np.argmax(sample_scores))

    return np.array(preds)


# ============================
# 6. SELECT THRESHOLD
# ============================

# Start with recommended threshold (can adjust between 0.2–0.6)
UNKNOWN_THRESHOLD = 0.35

val_pred_unknown = predict_with_unknown(svm, scaler, X_val, UNKNOWN_THRESHOLD)

print("\nClassification Report (with Unknown class):")
print(classification_report(y_val, val_pred_unknown))


# ============================
# 7. SAVE EVERYTHING
# ============================

joblib.dump(svm, "svm_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
joblib.dump(UNKNOWN_THRESHOLD, "unknown_threshold.pkl")

print("\nSaved:")
print(" - svm_model.pkl")
print(" - svm_scaler.pkl")
print(" - unknown_threshold.pkl")
