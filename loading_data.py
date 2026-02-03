import os
import cv2

RAW_PATH = "Fruit_dataset/train1/"

def load_raw_images():
    X, y, paths = [], [], []

    # Automatically detect class folders
    class_names = sorted([
        d for d in os.listdir(RAW_PATH)
        if os.path.isdir(os.path.join(RAW_PATH, d))
    ])

    # Create label mapping dynamically
    class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_names)}


    for folder, label in class_to_label.items():
        folder_path = os.path.join(RAW_PATH, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)   # RAW BGR IMAGE

            if img is None:
                continue

            X.append(img)
            y.append(label)
            paths.append(img_path)

    return X, y, paths, class_to_label


X_raw, y_raw, raw_paths, class_map = load_raw_images()
print("Loaded RAW images:", len(X_raw))
