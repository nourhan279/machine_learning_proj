import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# 1. Update Imports
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model

def load_processed_images(path):
    X = []
    y = []

    for label in os.listdir(path):
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            # 2. LOAD AS COLOR (Remove cv2.IMREAD_GRAYSCALE)
            # MobileNetV2 was trained on color images; it needs 3 channels.
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize to the standard CNN size
            img = cv2.resize(img, (224, 224))

            # Ensure it is RGB (OpenCV loads BGR by default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            X.append(img)
            y.append(int(label))

    return np.array(X), np.array(y)

# Load data
X_all, y_all = load_processed_images("data/processed/train/")
print("Loaded images:", X_all.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# 3. INITIALIZE MOBILENET V2
# 'pooling=avg' automatically converts the last conv layer into a 1280-dim vector
cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,    # Removes the 1000-class head
    pooling='avg',         # Adds Global Average Pooling (result: 1280 features)
    input_shape=(224, 224, 3)
)

def extract_cnn_features(model, X):
    # Use the MobileNetV2 specific preprocessing
    X = preprocess_input(X.astype("float32"))
    # MobileNetV2 is much faster at prediction
    features = model.predict(X, batch_size=32, verbose=1)
    return features

print("Extracting MobileNetV2 features...")
X_train_cnn = extract_cnn_features(cnn_model, X_train)
X_val_cnn   = extract_cnn_features(cnn_model, X_val)

# Verify the new shape (Should be 1280 instead of 4096)
print("New feature shape:", X_train_cnn.shape)

# Save results
os.makedirs("data/MobileNet_svm_features", exist_ok=True)
joblib.dump(X_train_cnn, "data/MobileNet_svm_features/train_features.pkl")
joblib.dump(y_train,     "data/MobileNet_svm_features/train_labels.pkl")
joblib.dump(X_val_cnn,   "data/MobileNet_svm_features/val_features.pkl")
joblib.dump(y_val,       "data/MobileNet_svm_features/val_labels.pkl")
print("MobileNetV2 feature files saved successfully!")