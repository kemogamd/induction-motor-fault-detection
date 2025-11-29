import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# LOAD DATA
# =====================================================
print("Loading CNN data...")
X_train = np.load("X_train_norm.npy")
X_test = np.load("X_test_norm.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# CNN needs (samples, timesteps, channels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Reshaped for CNN:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# =====================================================
# BUILD 1D CNN MODEL
# =====================================================
model = Sequential([
    Conv1D(32, 7, activation="relu", padding="same",
           input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),

    Conv1D(64, 5, activation="relu", padding="same"),
    MaxPooling1D(2),

    Conv1D(128, 3, activation="relu", padding="same"),
    GlobalMaxPooling1D(),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(5, activation="softmax")  # 5 classes
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# =====================================================
# TRAIN
# =====================================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

model.save("cnn_model.h5")
print("CNN model saved as cnn_model.h5")

# =====================================================
# EVALUATE
# =====================================================
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# =====================================================
# TRAINING CURVES
# =====================================================
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.show()

# =====================================================
# SAVE EXTENDED CNN TRAINING CURVES FOR APPENDIX D
# =====================================================

print("Saving extended CNN training curves...")

# Accuracy curve
plt.figure(figsize=(7, 4))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("CNN Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("cnn_acc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# Loss curve
plt.figure(figsize=(7, 4))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("CNN Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("cnn_loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()

print("Saved cnn_acc_curve.png and cnn_loss_curve.png")
