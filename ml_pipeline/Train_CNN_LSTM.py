import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# LOAD NORMALIZED DATA
# =====================================================
print("Loading normalized data for CNN-LSTM...")

X_train = np.load("X_train_norm.npy")
X_test = np.load("X_test_norm.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Shapes before reshape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# CNN/LSTM needs (samples, timesteps, channels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Shapes after reshape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

num_classes = len(np.unique(y_train))

# =====================================================
# BUILD CNN-LSTM MODEL
# =====================================================
print("\nBuilding CNN-LSTM model...")

model = Sequential([
    Conv1D(32, 7, activation="relu", padding="same",
           input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),

    Conv1D(64, 5, activation="relu", padding="same"),
    MaxPooling1D(2),

    LSTM(64, return_sequences=False),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =====================================================
# TRAIN
# =====================================================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

model.save("cnn_lstm_model.h5")
print("CNN-LSTM model saved as cnn_lstm_model.h5")

# =====================================================
# EVALUATE
# =====================================================
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\nClassification Report (CNN-LSTM):")
print(classification_report(y_test, y_pred_labels))

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("CNN-LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("cnn_lstm_confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# TRAINING CURVES
# =====================================================
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("CNN-LSTM Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("CNN-LSTM Loss")
plt.legend()

plt.tight_layout()
plt.savefig("cnn_lstm_training_curves.png", dpi=300)
plt.show()

print("Saved cnn_lstm_confusion_matrix.png and cnn_lstm_training_curves.png")
