import os
import cv2
import numpy as np 
from loading_data import CLASS_NAMES, RAW_PATH
from save_augmented_data import AUG_PATH

PROCESSED_PATH = "data/processed/"

def preprocess_image(img, size=(224, 224)):

    # 1. Resize to 224x224
    img = cv2.resize(img, size)

    # If the image is accidentally grayscale, convert it to BGR/RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    return img

def preprocess_and_save():

    # Process original raw dataset
    for folder, label in CLASS_NAMES.items():
        folder_path = os.path.join(RAW_PATH, folder)
        if not os.path.exists(folder_path): continue
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            proc = preprocess_image(img)
            
            out_folder = os.path.join(PROCESSED_PATH, "train", str(label))
            os.makedirs(out_folder, exist_ok=True)
            
            # Save as standard uint8 BGR image (3 channels)
            cv2.imwrite(os.path.join(out_folder, img_name), proc)

    # Process augmented dataset
    if os.path.exists(AUG_PATH):
        for label in os.listdir(AUG_PATH):
            aug_label_path = os.path.join(AUG_PATH, label)
            if not os.path.isdir(aug_label_path): continue
            
            for img_name in os.listdir(aug_label_path):
                img_path = os.path.join(aug_label_path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue

                proc = preprocess_image(img)

                out_folder = os.path.join(PROCESSED_PATH, "train", label)
                os.makedirs(out_folder, exist_ok=True)
                
                # Save as standard uint8 BGR image (3 channels)
                cv2.imwrite(os.path.join(out_folder, img_name), proc)

    print("Processed images saved successfully!")

preprocess_and_save()

# print the number of processed images in total
total_processed = 0
train_path = os.path.join(PROCESSED_PATH, "train")
if os.path.exists(train_path):
    for label in os.listdir(train_path):
        label_folder = os.path.join(train_path, label)
        total_processed += len(os.listdir(label_folder))

print("Total processed images saved:", total_processed)