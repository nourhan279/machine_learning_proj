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
print("\nValidation Accuracy (no Unknown):", acc)
print(classification_report(y_val, val_pred))


#save the models
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")


print("\nSaved:")
print("knn_model.pkl")
print("knn_scaler.pkl")






