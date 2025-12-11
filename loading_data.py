import os
import cv2
import numpy as np

RAW_PATH = "dataset/"

CLASS_NAMES = {
    "Glass": 0,
    "Paper": 1,
    "Cardboard": 2,
    "Plastic": 3,
    "Metal": 4,
    "Trash": 5
}

def load_raw_images():
    X = []
    y = []
    paths = []

    for folder, label in CLASS_NAMES.items():
        folder_path = os.path.join(RAW_PATH, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)   # RAW RGB IMAGE

            if img is None:
                continue

            X.append(img)
            y.append(label)
            paths.append(img_path)

    return X, y, paths

X_raw, y_raw, raw_paths = load_raw_images()
print("Loaded RAW images:", len(X_raw))
