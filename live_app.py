import joblib
import numpy as np
import cv2
import time
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# ============================
# 1. INITIALIZATION & LOADING
# ============================

print("Loading trained model components...")
try:
    # Make sure these filenames match exactly what you saved in svm_cnn.py
    svm = joblib.load("models/svm_mobilenet_model.pkl") 
    scaler = joblib.load("models/mobilenet_scaler.pkl")
    print("Models and Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found in 'models/' folder.")
    exit()

CLASS_LABELS = {
    0: "Glass", 1: "Paper", 2: "Cardboard", 
    3: "Plastic", 4: "Metal", 5: "Trash", 6: "Unknown"
}

IMAGE_SIZE = (224, 224) 

# --- Load MobileNetV2 Feature Extractor ---
print("Loading MobileNetV2 feature extractor...")
# pooling='avg' ensures we get the 1280-dim feature vector automatically
cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

# ============================
# 2. CORE PREDICTION FUNCTION
# ============================

def classify_frame(frame_features):
    X_sample = frame_features.reshape(1, -1)
    X_scaled = scaler.transform(X_sample)
    
    # Use decision_function for confidence scores
    scores = svm.decision_function(X_scaled)[0] 
    max_score = np.max(scores)
    predicted_class_index = np.argmax(scores)
    
    # Threshold logic (0.6 is a good starting point for RBF SVM)
    if max_score < 0.6:
        final_prediction_index = 6 
    else:
        final_prediction_index = predicted_class_index

    return CLASS_LABELS[final_prediction_index], max_score

# ============================
# 3. MAIN REAL-TIME LOOP
# ============================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_count = 0
start_time = time.time()
classification_frame_count = 0 
last_label = "Waiting..."
last_confidence = 0.0

print("Starting video feed. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    classification_frame_count += 1

    # Logic: Only run the heavy CNN every 5th frame to keep FPS high
    if classification_frame_count % 5 == 0:
        # 1. Resize
        processed_frame = cv2.resize(frame, IMAGE_SIZE) 
        
        # 2. Prepare for Keras (1, 224, 224, 3)
        # MobileNetV2 needs BGR to RGB conversion for best accuracy
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        X_sample = np.expand_dims(rgb_frame, axis=0).astype("float32")

        # 3. MobileNetV2 Specific Preprocessing
        X_processed = preprocess_input(X_sample)

        # 4. Extract 1280-dim Features
        features = cnn_model.predict(X_processed, verbose=0)[0]
        
        # 5. SVM Prediction
        last_label, last_confidence = classify_frame(features)
        
    # --- Visualization ---
    text = f"Pred: {last_label} (Conf: {last_confidence:.2f})"
    color = (0, 255, 0) if last_label != "Unknown" else (0, 165, 255)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # FPS Logic
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        frame_count = 0
        start_time = time.time()

    cv2.imshow('Real-Time Waste Classifier (MobileNetV2)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()