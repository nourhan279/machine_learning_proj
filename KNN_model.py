import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


#load vector data from hog
X_train = joblib.load("data/hog/hog_train_features.pkl")
y_train = joblib.load("data/hog/hog_train_labels.pkl")
X_val = joblib.load("data/hog/hog_val_features.pkl")
y_val = joblib.load("data/hog/hog_val_labels.pkl")

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
# Create 5 samples of random noise to represent truly Unknown items for test only
fake_unknown_images = np.random.rand(5, 8100) 

# Combine validation data with fake one
X_test_combined = np.vstack([X_val, fake_unknown_images])
y_test_combined = np.concatenate([y_val, np.full(5, 6)])

print(f"\nCreated combined test set: {X_test_combined.shape[0]} total samples.")


#unknown class detection
def predict_with_unknown_knn(model, X, distance_threshold):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
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
distances, index = knn.kneighbors(X_val_scaled, n_neighbors=1)
print("Distance(Known Val Set) min, mean, max:", distances.min(), distances.mean(), distances.max())
#choose threshold above the maximum distance (127) by small value 
DISTANCE_THRESHOLD = 130


#prediction
val_pred_unknown_combined = predict_with_unknown_knn(knn,X_test_combined, DISTANCE_THRESHOLD)
acc2=accuracy_score(y_test_combined,val_pred_unknown_combined)
print("\n Validation Accuracy (including Unknown class):", acc2)
class_names2 = ['Glass( 0)', 'Paper (1)', 'Cardboard (2)', 'Plastic (3)', 'Metal (4)', 'Trash (5)', 'Unknown (6)']
print(classification_report(y_test_combined, val_pred_unknown_combined, target_names=class_names2, zero_division=0)) 


#save the models
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")
joblib.dump(DISTANCE_THRESHOLD, "knn_unknown_threshold.pkl")

print("\nSaved:")
print("knn_model.pkl")
print("knn_scaler.pkl")
print("knn_unknown_threshold.pkl")

