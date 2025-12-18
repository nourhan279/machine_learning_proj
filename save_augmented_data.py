import os
import random
import cv2
import numpy as np
import loading_data as ld

AUG_PATH = "data/augmented/"

def rotate(img):
    h, w = img.shape[:2]
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def flip(img):
    return cv2.flip(img, 1)

def brightness(img):
    factor = random.uniform(0.7, 1.3)
    img_new = img * factor
    return np.clip(img_new, 0, 255).astype(np.uint8)

def zoom(img):
    h, w = img.shape[:2]
    factor = random.uniform(0.85, 1.0)
    nh, nw = int(h*factor), int(w*factor)
    top = (h - nh) // 2
    left = (w - nw) // 2
    cropped = img[top:top+nh, left:left+nw]
    return cv2.resize(cropped, (w, h))

def save_augmented_images(X, y, percentage=0.6):
    total_new = int(len(X) * percentage)

    for i in range(total_new):
        idx = random.randint(0, len(X) - 1)
        original = X[idx]
        label = y[idx]

        aug_fn = random.choice([rotate, flip, brightness, zoom])
        aug_img = aug_fn(original)

        label_folder = os.path.join(AUG_PATH, str(label))
        os.makedirs(label_folder, exist_ok=True)

        file_path = os.path.join(label_folder, f"aug_{i}.jpg")
        cv2.imwrite(file_path, aug_img)

    print("Augmentation complete!")

save_augmented_images(ld.X_raw, ld.y_raw, percentage=0.3)


# print the number of augmented images in total
total_augmented = 0

for label in os.listdir(AUG_PATH):
    label_folder = os.path.join(AUG_PATH, label)
    total_augmented += len(os.listdir(label_folder))

print("Total augmented images saved:", total_augmented)