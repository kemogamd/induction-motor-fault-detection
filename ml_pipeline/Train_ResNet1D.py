import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     Activation, Add, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# LOAD NORMALIZED DATA
# =====================================================
print("Loading normalized data for ResNet1D...")

X_train = np.load("X_train_norm.npy")
X_test = np.load("X_test_norm.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Shapes before reshape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Reshape for Conv1D: (samples, timesteps, channels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Shapes after reshape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

num_classes = len(np.unique(y_train))
input_shape = (X_train.shape[1], 1)

# =====================================================
# RESNET1D ARCHITECTURE
# =====================================================


def residual_block(x, filters, kernel_size=7, stride=1):
    shortcut = x

    x = Conv1D(filters, kernel_size, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(filters, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    # Match dimensions if stride changes
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


print("\nBuilding ResNet1D model...")

inputs = Input(shape=input_shape)

x = Conv1D(32, 7, padding="same", strides=1)(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = residual_block(x, 32, kernel_size=7, stride=1)
x = residual_block(x, 64, kernel_size=5, stride=2)
x = residual_block(x, 64, kernel_size=5, stride=1)
x = residual_block(x, 128, kernel_size=3, stride=2)

x = GlobalAveragePooling1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
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

model.save("resnet1d_model.h5")
print("ResNet1D model saved as resnet1d_model.h5")

# =====================================================
# EVALUATE
# =====================================================
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\nClassification Report (ResNet1D):")
print(classification_report(y_test, y_pred_labels))

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("ResNet1D Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("resnet1d_confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# TRAINING CURVES
# =====================================================
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("ResNet1D Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("ResNet1D Loss")
plt.legend()

plt.tight_layout()
plt.savefig("resnet1d_training_curves.png", dpi=300)
plt.show()

print("Saved resnet1d_confusion_matrix.png and resnet1d_training_curves.png")
