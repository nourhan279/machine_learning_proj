import os
import cv2

from loading_data import CLASS_NAMES, RAW_PATH
from save_augmented_data import AUG_PATH


PROCESSED_PATH = "data/processed/"

def preprocess_image(img, size=(128,128)):
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    return img

def preprocess_and_save():
    # Load original
    X = []
    y = []

    # Process original raw dataset
    for folder, label in CLASS_NAMES.items():
        folder_path = os.path.join(RAW_PATH, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            proc = preprocess_image(img)
            X.append(proc)
            y.append(label)

            out_folder = os.path.join(PROCESSED_PATH, "train", str(label))
            os.makedirs(out_folder, exist_ok=True)
            cv2.imwrite(os.path.join(out_folder, img_name), (proc*255).astype("uint8"))

    # Process augmented dataset
    for label in os.listdir(AUG_PATH):
        aug_label_path = os.path.join(AUG_PATH, label)
        for img_name in os.listdir(aug_label_path):
            img_path = os.path.join(aug_label_path, img_name)
            img = cv2.imread(img_path)

            proc = preprocess_image(img)
            X.append(proc)
            y.append(int(label))

            out_folder = os.path.join(PROCESSED_PATH, "train", label)
            os.makedirs(out_folder, exist_ok=True)
            cv2.imwrite(os.path.join(out_folder, img_name), (proc*255).astype("uint8"))

    print("Processed images saved successfully!")

preprocess_and_save()

# print the number of processed images in total
total_processed = 0
for label in os.listdir(os.path.join(PROCESSED_PATH, "train")):
    label_folder = os.path.join(PROCESSED_PATH, "train", label)
    total_processed += len(os.listdir(label_folder))

print("Total processed images saved:", total_processed)
