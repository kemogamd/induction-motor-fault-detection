import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import joblib

# =====================================================
# LOAD FFT FEATURES
# =====================================================
print("Loading FFT data for Random Forest...")

X_train = np.load("X_train_fft.npy")
X_test = np.load("X_test_fft.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# =====================================================
# TRAIN RANDOM FOREST CLASSIFIER
# =====================================================
print("\nTraining Random Forest classifier...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample"
)

rf.fit(X_train, y_train)

joblib.dump(rf, "rf_model.pkl")
print("Saved rf_model.pkl")

# =====================================================
# EVALUATE
# =====================================================
y_pred = rf.predict(X_test)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================
print("\nGenerating Random Forest learning curve...")


train_sizes, train_scores, val_scores = learning_curve(
    rf,
    X_train,
    y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Training accuracy")
plt.plot(train_sizes, val_mean, "o-", label="Validation accuracy")
plt.title("Random Forest Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rf_learning_curve.png", dpi=300)
plt.show()

print("Saved rf_confusion_matrix.png and rf_learning_curve.png")
