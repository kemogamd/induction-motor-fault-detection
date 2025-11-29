import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# =====================================================
# LOAD FFT FEATURES
# =====================================================
print("Loading FFT data for XGBoost...")

X_train = np.load("X_train_fft.npy")
X_test = np.load("X_test_fft.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

num_classes = len(np.unique(y_train))
print("Detected number of classes:", num_classes)

# =====================================================
# TRAIN XGBOOST CLASSIFIER
# =====================================================
print("\nTraining XGBoost classifier...")

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)

joblib.dump(xgb, "xgb_model.pkl")
print("Saved xgb_model.pkl")

# =====================================================
# EVALUATE
# =====================================================
y_pred = xgb.predict(X_test)

print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# FEATURE IMPORTANCE PLOT
# =====================================================
print("Plotting XGBoost feature importances...")

importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1][:30]  # Top 30 bins

plt.figure(figsize=(8, 5))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), indices, rotation=90)
plt.title("XGBoost Top 30 Feature Importances (FFT bins)")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=300)
plt.show()

print("Saved xgb_confusion_matrix.png and xgb_feature_importance.png")
