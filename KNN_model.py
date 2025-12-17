import joblib
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score


#load vector data from cnn
FEATURE_PATH = "data/MobileNet_features"

X_train = joblib.load(os.path.join(FEATURE_PATH, "train_features.pkl"))
y_train = joblib.load(os.path.join(FEATURE_PATH, "train_labels.pkl"))
X_val = joblib.load(os.path.join(FEATURE_PATH, "val_features.pkl"))
y_val = joblib.load(os.path.join(FEATURE_PATH, "val_labels.pkl"))
print("Train features:",X_train.shape)
print("Validation features:",X_val.shape)


#Feacture Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


#train model
knn = KNeighborsClassifier(
    n_neighbors=2, #better acc found
    weights='distance',     
    metric='euclidean'    
)
knn.fit(X_train_scaled, y_train)
print("KNN training complete")

#Normal (no unknown class)
val_pred = knn.predict(X_val_scaled)
acc = accuracy_score(y_val, val_pred)
class_names = ['Glass (0)', 'Paper (1)', 'Cardboard (2)', 'Plastic (3)', 'Metal (4)', 'Trash (5)']
print("\nValidation Accuracy (no Unknown):", acc)
print(classification_report(y_val, val_pred,target_names=class_names, zero_division=0))


#---------------------------------------- unknown class ------------------------------------------------
#unknown class detection
def predict_with_unknown_knn(model,scaler, X, distance_threshold):
    X_scaled = scaler.transform(X)
    # Calculate distance to the single nearest neighbor
    distances, index= model.kneighbors(X_scaled, n_neighbors=1)
    preds = []
    for i in range(len(distances)):
        if distances[i][0] > distance_threshold:
            preds.append(6) 
        else:
            preds.append(model.predict([X_scaled[i]])[0]) 
    return np.array(preds)


# Use the original scaled validation data to check distance values
distances, _ = knn.kneighbors(X_val_scaled, n_neighbors=1)
print("Distance(Known Val Set) min, mean, max:", distances.min(), distances.mean(), distances.max())
#increase from max little bit
DISTANCE_THRESHOLD = np.max(distances) * 1.1 

labels_to_show = [0, 1, 2, 3, 4, 5, 6]

#prediction
val_pred_unknown_combined = predict_with_unknown_knn(knn,scaler,X_val, DISTANCE_THRESHOLD)
acc2=accuracy_score(y_val,val_pred_unknown_combined)
print("\n Validation Accuracy (including Unknown class):", acc2)
class_names2 = ['Glass( 0)', 'Paper (1)', 'Cardboard (2)', 'Plastic (3)', 'Metal (4)', 'Trash (5)', 'Unknown (6)']
print(classification_report(y_val, val_pred_unknown_combined,labels=labels_to_show, target_names=class_names2, zero_division=0)) 


#save the models
joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(scaler, "models/knn_scaler.pkl")
joblib.dump(DISTANCE_THRESHOLD, "models/knn_unknown_threshold.pkl")

print("\nSaved:")
print("knn_model.pkl")
print("knn_scaler.pkl")
print("knn_unknown_threshold.pkl")

