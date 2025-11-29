import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import joblib

# =====================================================
# LOAD FFT FEATURES
# =====================================================
print("Loading FFT data for MLP...")

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
# TRAIN MLP CLASSIFIER
# =====================================================
print("\nTraining MLP classifier...")

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=64,
    learning_rate="adaptive",
    max_iter=200,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=10,
    verbose=True
)

mlp.fit(X_train, y_train)

joblib.dump(mlp, "mlp_model.pkl")
print("Saved mlp_model.pkl")

# =====================================================
# EVALUATE
# =====================================================
y_pred = mlp.predict(X_test)

print("\nClassification Report (MLP):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("MLP Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("mlp_confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================
print("\nGenerating MLP learning curve...")

train_sizes, train_scores, val_scores = learning_curve(
    MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=200,
        random_state=42
    ),
    X_train,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Training accuracy")
plt.plot(train_sizes, val_mean, "o-", label="Validation accuracy")
plt.title("MLP Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mlp_learning_curve.png", dpi=300)
plt.show()

print("Saved mlp_confusion_matrix.png and mlp_learning_curve.png")
