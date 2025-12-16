import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16, preprocess_input
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

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize for CNN
            img = cv2.resize(img, (224, 224))

            # Convert grayscale → RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            X.append(img)
            y.append(int(label))

    return np.array(X), np.array(y)


X_all, y_all = load_processed_images("data/processed/train/")
print("Loaded images:", X_all.shape)


X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)


base_model = VGG16(
    weights="imagenet",
    include_top=True
)

# Remove classification layer, keep features
cnn_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("fc2").output  # 4096-dim features
)


def extract_cnn_features(model, X):
    X = preprocess_input(X.astype("float32"))
    features = model.predict(X, batch_size=16, verbose=1)
    return features


print("Extracting CNN features...")

X_train_cnn = extract_cnn_features(cnn_model, X_train)
X_val_cnn   = extract_cnn_features(cnn_model, X_val)

print("CNN train features:", X_train_cnn.shape)
print("CNN val features:", X_val_cnn.shape)


os.makedirs("data/cnn_features", exist_ok=True)

joblib.dump(X_train_cnn, "data/cnn_features/train_features.pkl")
joblib.dump(y_train,     "data/cnn_features/train_labels.pkl")
joblib.dump(X_val_cnn,   "data/cnn_features/val_features.pkl")
joblib.dump(y_val,       "data/cnn_features/val_labels.pkl")

print("CNN feature files saved successfully!")
