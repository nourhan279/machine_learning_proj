import joblib
import numpy as np
import cv2
import time
from skimage.feature import hog 

# ============================
# 1. INITIALIZATION & LOADING
# ============================

# --- A. Load Trained Components ---
print("Loading trained model components...")
try:
    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("svm_scaler.pkl")
    UNKNOWN_THRESHOLD = joblib.load("unknown_threshold.pkl")
    print(f"Models loaded successfully. Unknown Threshold: {UNKNOWN_THRESHOLD}")
except FileNotFoundError:
    print("Error: Model files not found. Ensure 'svm_model.pkl', 'svm_scaler.pkl', and 'unknown_threshold.pkl' are available.")
    exit()

# --- B. Define Classes ---

CLASS_LABELS = {
    0: "Glass",
    1: "Paper",
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
    6: "Unknown"  # Index 6 is reserved for the Unknown/Low Confidence class
}

# --- C. Define Image & HOG Parameters  ---
IMAGE_SIZE = (128, 128) # As in preprocessing.py
HOG_PARAMS = { # As in feature_extraction.py
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False,
    'feature_vector': True
}


# ============================
# 2. CORE PREDICTION FUNCTION
# ============================

def classify_frame(frame_features):
    """
    Classifies a single frame's features using the SVM, scaler, and 
    unknown class threshold logic.
    """
    
    # Reshape the feature vector from (N,) to (1, N) for the scaler/model
    X_sample = frame_features.reshape(1, -1)
    
    # 1. Scaling (Must use the trained scaler)
    X_scaled = scaler.transform(X_sample)
    
    # 2. Get Decision Function Scores (used for confidence/unknown class)
    # The decision_function returns an array of shape (1, n_classes), we grab the scores.
    scores = svm.decision_function(X_scaled)[0] 
    
    # 3. Find Max Score and Prediction
    max_score = np.max(scores)
    predicted_class_index = np.argmax(scores)
    
    # 4. Apply Unknown Threshold Logic
    if max_score < UNKNOWN_THRESHOLD:
        final_prediction_index = 6 # Unknown class
    else:
        final_prediction_index = predicted_class_index

    # Convert index to meaningful label
    label = CLASS_LABELS[final_prediction_index]
    
    # The max score serves as a confidence metric for display
    return label, max_score


# ============================
# 3. MAIN REAL-TIME LOOP
# ============================

cap = cv2.VideoCapture(0) # Open the default camera 

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Variables for FPS calculation
frame_count = 0
start_time = time.time()

print("Starting video feed. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------------------------------
    # --- A. PRE-PROCESSING & HOG EXTRACTION (Matches training Pipeline) ---
    # -----------------------------------------------------------------
    
    # 1. Resize
    processed_frame = cv2.resize(frame, IMAGE_SIZE) 
    
    # 2. Convert to Grayscale
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    # 3. Normalize (REQUIRED for HOG/SVM as per training code)
    normalized_frame = gray_frame.astype("float32") / 255.0

    # 4. Extract HOG Features
    features = hog(normalized_frame, **HOG_PARAMS)
    
    # --- B. Real-Time Prediction ---
    label, confidence = classify_frame(features)
    
    # --- C. Visualization and Output ---
    
    # Display Prediction and Confidence
    text = f"Pred: {label} (Conf: {confidence:.2f})"
    color = (0, 255, 0) if label != "Unknown" else (0, 165, 255) # Green for Known, Orange for Unknown
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # FPS Calculation and Display
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    if elapsed_time > 1.0: # Update FPS every second
        fps = frame_count / elapsed_time
        fps_text = f"FPS: {fps:.2f}"
        # Display FPS in red
        cv2.putText(frame, fps_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) 
        frame_count = 0
        start_time = time.time()

    cv2.imshow('Real-Time SVM Classifier (FPS Check)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()