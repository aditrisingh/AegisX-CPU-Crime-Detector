import os
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ====== 📂 Load features from directory ======
def load_features_from_folder(folder_path, label):
    features = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            feature = np.load(os.path.join(folder_path, filename))
            features.append(feature)
            labels.append(label)
    return features, labels

# 🔍 Load crime & no-crime features
crime_features, crime_labels = load_features_from_folder("last hope/Test Features/Crime", 1)
no_crime_features, no_crime_labels = load_features_from_folder("last hope/Test Features/No Crime", 0)

# 🔗 Combine features & labels
X = np.array(crime_features + no_crime_features)
y = np.array(crime_labels + no_crime_labels)

# ====== 📊 Load Scaler + Classifier ======
scaler = joblib.load("scaler2.joblib")
clf = joblib.load("crime_detector2.joblib")

# ====== 🔮 Prediction ======
X_scaled = scaler.transform(X)
y_pred = clf.predict(X_scaled)
y_prob = clf.predict_proba(X_scaled)[:, 1]

# ====== 📈 Metrics ======
accuracy = accuracy_score(y, y_pred)
print(f"\n✅ Accuracy on Test Set: {accuracy * 100:.2f}%\n")
print("📋 Classification Report:\n")
print(classification_report(y, y_pred, target_names=["No Crime", "Crime"]))

# ====== 💾 Save Predictions to CSV ======
results_df = pd.DataFrame({
    "True Label": y,
    "Predicted Label": y_pred,
    "Probability": y_prob
})
results_df.to_csv("predictions_report.csv", index=False)
print("📁 Predictions saved to: predictions_report.csv")

# ====== 🎨 Confusion Matrix ======
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["No Crime", "Crime"], yticklabels=["No Crime", "Crime"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🧠 Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ====== 🧪 ROC Curve ======
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="teal", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📊 ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# ====== 💖 Precision-Recall Curve ======
precision, recall, _ = precision_recall_curve(y, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="crimson", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("🎯 Precision-Recall Curve")
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.show()
