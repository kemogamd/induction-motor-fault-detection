import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Loading raw dataset...")
X = np.load("signals.npy")   # (500, 12800)
y = np.load("labels.npy")    # (500,)
print("Loaded:", X.shape, y.shape)

# =====================================================
# 1. SPLIT RAW DATA FIRST (BEFORE NORMALIZATION)
# =====================================================
print("\nSplitting dataset BEFORE normalization...")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train_raw.shape)
print("Test :", X_test_raw.shape)

# =====================================================
# 2. SAFE NORMALIZATION (fit ONLY on training set)
# =====================================================
print("\nNormalizing...")

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_raw)
X_test_norm = scaler.transform(X_test_raw)

print("Normalized shapes:", X_train_norm.shape, X_test_norm.shape)

# =====================================================
# 3. SAFE FFT FEATURE EXTRACTION (train & test separately)
# =====================================================


def compute_fft(x):
    # real FFT magnitude
    fft_vals = np.abs(np.fft.rfft(x))
    return fft_vals[:500]    # first 500 bins


print("\nComputing FFT (train)...")
X_train_fft = np.array([compute_fft(x) for x in X_train_norm])

print("Computing FFT (test)...")
X_test_fft = np.array([compute_fft(x) for x in X_test_norm])

print("FFT shapes:", X_train_fft.shape, X_test_fft.shape)

# =====================================================
# 4. SAVE CLEAN DATASETS
# =====================================================
np.save("X_train_norm.npy", X_train_norm)
np.save("X_test_norm.npy", X_test_norm)
np.save("X_train_fft.npy", X_train_fft)
np.save("X_test_fft.npy", X_test_fft)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nSaved clean preprocessed dataset:")
print(" - X_train_norm.npy")
print(" - X_test_norm.npy")
print(" - X_train_fft.npy")
print(" - X_test_fft.npy")
print(" - y_train.npy")
print(" - y_test.npy")

print("\n🎉 Preprocessing complete — no leakage!")
