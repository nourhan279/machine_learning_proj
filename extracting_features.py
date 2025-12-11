import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split


# ===========================================================
# 1) Load processed images (grayscale + resized)
# ===========================================================

def load_processed_images(path):
    X = []
    y = []

    for label in os.listdir(path):
        folder = os.path.join(path, label)

        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = img.astype("float32") / 255.0
            X.append(img)
            y.append(int(label))

    return np.array(X), np.array(y)


X_all, y_all = load_processed_images("data/processed/train/")
print("Loaded processed images:", X_all.shape)


# ===========================================================
# 2) Split into TRAIN and VALIDATION sets
# ===========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

print("Train set:", X_train.shape)
print("Val set:", X_val.shape)


# ===========================================================
# 3) Extract HOG features
# ===========================================================

def extract_hog(X):
    features = []
    for img in X:
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(feat)
    return np.array(features)


print("Extracting HOG...")

X_train_hog = extract_hog(X_train)
X_val_hog   = extract_hog(X_val)

print("HOG train shape:", X_train_hog.shape)
print("HOG val shape:", X_val_hog.shape)


# ===========================================================
# 4) SAVE FEATURE FILES FOR SVM
# ===========================================================

os.makedirs("data/hog", exist_ok=True)

joblib.dump(X_train_hog, "data/hog/hog_train_features.pkl")
joblib.dump(y_train,     "data/hog/hog_train_labels.pkl")
joblib.dump(X_val_hog,   "data/hog/hog_val_features.pkl")
joblib.dump(y_val,       "data/hog/hog_val_labels.pkl")

print("HOG feature files saved successfully!")
