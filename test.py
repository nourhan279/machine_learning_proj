import os
import cv2
import numpy as np
import joblib
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

def predict(dataFilePath, bestModelPath):
    
    predictions = []
    
    # 1. Load the Model & Scaler
    try:
        svm_model = joblib.load(bestModelPath)
        scaler = joblib.load("models/mobilenet_scaler.pkl")
    except Exception as e:
        return f"Error loading models: {e}"

    # 2. Initialize the Feature Extractor (MobileNetV2)
    cnn_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )

  
    # 3. Process Images in the Folder
    if not os.path.exists(dataFilePath):
        return "Folder not found."

    for img_name in os.listdir(dataFilePath):
        img_path = os.path.join(dataFilePath, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # --- PREPROCESSING ---
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare for Keras (adds the batch dimension)
        X = np.expand_dims(img, axis=0).astype("float32")
        X_preprocessed = preprocess_input(X)

        # --- FEATURE EXTRACTION ---
        features = cnn_model.predict(X_preprocessed, verbose=0) # Result: (1, 1280)

        # --- SVM INFERENCE ---
        features_scaled = scaler.transform(features)
        scores = svm_model.decision_function(features_scaled)[0]
        
        predicted_idx = np.argmax(scores)
        max_score = scores[predicted_idx]

        # Apply threshold
        if max_score < 0.6:
            predictions.append(6) # Unknown
        else:
            predictions.append(int(predicted_idx))

    return predictions

results = predict("dataFilePath/testml", "models/svm_mobilenet_model.pkl")
print(results)