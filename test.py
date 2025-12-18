import os
import cv2
import numpy as np
import joblib
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

CLASS_NAMES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]

def predict(dataFilePath, bestModelPath):
    
    predictions = []
    threshold = 4
    
    # 1. Load the Model & Scaler
    try:
        best_model = joblib.load(bestModelPath)
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

        # Feature Extraction
        features = cnn_model.predict(X_preprocessed, verbose=0) # Result: (1, 1280)

        # Scale Features & Predict
        features_scaled = scaler.transform(features)
        scores = best_model.decision_function(features_scaled)
        
        # Get confidence and predicted class
        confidence = np.max(scores, axis=1)[0]
        pred_class = best_model.predict(features_scaled)[0]

        # Determine final label with unknown class handling
        label = (
            "Unknown"
            if confidence < threshold
            else CLASS_NAMES[pred_class]
        )

        predictions.append(label)

    return predictions

results = predict("dataFilePath/testml", "models/svm_mobilenet_model.pkl")
print(results)