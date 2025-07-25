import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from joblib import dump

# --- Paths ---
base_path = "last hope/Output_IR3"
crime_path = os.path.join(base_path, "Crime")
no_crime_path = os.path.join(base_path, "No Crime")

# --- Load Features ---
X = []
y = []

for file in os.listdir(crime_path):
    if file.endswith(".npy"):
        feat = np.load(os.path.join(crime_path, file))
        X.append(feat)
        y.append(1)

for file in os.listdir(no_crime_path):
    if file.endswith(".npy"):
        feat = np.load(os.path.join(no_crime_path, file))
        X.append(feat)
        y.append(0)

X = np.array(X)
y = np.array(y)

# --- Flatten ---
if len(X.shape) > 2:
    X = X.reshape((X.shape[0], -1))

# --- Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Hyperparameter Tuning ---
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

print("🔍 Performing GridSearchCV on entire dataset...")
grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=1, n_jobs=-1)
grid.fit(X_scaled, y)

print(f"🏆 Best Parameters: {grid.best_params_}")
print(f"✅ Cross-validated Accuracy: {grid.best_score_:.4f}")

# --- Final Training on Entire Data with Best Params ---
best_clf = grid.best_estimator_
best_clf.fit(X_scaled, y)

# --- Save Model ---
dump(best_clf, "crime_detector2.joblib")
dump(scaler, "scaler2.joblib")

print("✅ Final SVM model trained on full dataset and saved.")
