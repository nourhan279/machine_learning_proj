import os
import cv2
import numpy as np
import joblib
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import pandas as pd

def predict(bestModelPath):
    dataFilePath = input("Enter File Path: ")
    
    predictions = []

    CLASS_NAMES = [
    "Glass",
    "Paper",
    "Cardboard",
    "Plastic",
    "Metal",
    "Trash"
    ]
    
    # 1. Load the Model 
    try:
        
        metadata = joblib.load(bestModelPath)

        # Extract components from the dictionary
        model = metadata["model"]
        scaler = metadata["scaler"]
        threshold = metadata["threshold"]
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
        scores = model.decision_function(features_scaled)
         
        # Get confidence and predicted class
        confidence = np.max(scores, axis=1)[0]
        pred_class = model.predict(features_scaled)[0]
        
        # Determine final label with unknown class handling
        if confidence < threshold:
            label = "Unknown"
        else:
            label = CLASS_NAMES[pred_class]
        predictions.append({"ImageName": img_name, "predictedlabel": label})

    return predictions

results = predict("models/final_svm_package.pkl")

try:
    df = pd.DataFrame(results)
    df.to_excel("ml_output.xlsx", index=False)
    print("Results saved to ml_output.xlsx")
except Exception as e:
    print(f"Error saving results to Excel: {e}")
    