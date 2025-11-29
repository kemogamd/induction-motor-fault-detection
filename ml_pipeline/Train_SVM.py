import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =====================================================
# LOAD FFT FEATURES
# =====================================================
print("Loading FFT data...")

X_train = np.load("X_train_fft.npy")
X_test = np.load("X_test_fft.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# =====================================================
# TRAIN SVM CLASSIFIER
# =====================================================
print("\nTraining SVM classifier...")

svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train, y_train)

joblib.dump(svm, "svm_model.pkl")
print("Saved svm_model.pkl")


# =====================================================
# EVALUATE
# =====================================================
y_pred = svm.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# =====================================================
# TRAINING CURVES (Learning Curve)
# =====================================================
print("\nGenerating learning curve...")

train_sizes, train_scores, val_scores = learning_curve(
    SVC(kernel="rbf", C=10, gamma="scale"),
    X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', label="Training accuracy")
plt.plot(train_sizes, val_mean, 'o-', label="Validation accuracy")
plt.title("SVM Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================
# VALIDATION CURVE (Varying C parameter)
# =====================================================
print("\nGenerating validation curve...")

C_values = np.logspace(-2, 2, 5)

train_scores, val_scores = validation_curve(
    SVC(kernel="rbf", gamma="scale"),
    X_train, y_train,
    param_name="C",
    param_range=C_values,
    cv=5,
    scoring="accuracy"
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.semilogx(C_values, train_mean, 'o-', label="Training accuracy")
plt.semilogx(C_values, val_mean, 'o-', label="Validation accuracy")
plt.title("SVM Validation Curve (C parameter)")
plt.xlabel("C value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()


# =====================================================
# EXTRA METRICS FOR APPENDIX D (Precision/Recall/F1)
# =====================================================

print("\nGenerating SVM extra metrics...")

report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).iloc[:-1, :]  # Remove "accuracy" row

plt.figure(figsize=(8, 5))
sns.heatmap(df.T, annot=True, cmap="Blues", fmt=".2f")
plt.title("SVM Precision / Recall / F1-score")
plt.savefig("svm_extra_metrics.png", dpi=300, bbox_inches='tight')
plt.show()

print("Saved svm_extra_metrics.png")
