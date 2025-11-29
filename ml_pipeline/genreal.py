import numpy as np
import random

SAMPLES = 12800    # signal length
FS = 1280          # sampling rate (example)

# ============================================================
#  BASIC UTILITY FUNCTIONS
# ============================================================


def add_noise(x, noise_level=0.02):
    """Gaussian + high-frequency noise"""
    noise = np.random.normal(0, noise_level, len(x))
    hf_noise = 0.01 * np.random.randn(len(x))
    return x + noise + hf_noise


def add_drift(x, amplitude=0.02):
    """Low-frequency drift"""
    drift = amplitude * np.sin(np.linspace(0, 3*np.pi, len(x)))
    return x + drift


def random_shift(x, max_shift=300):
    """Random circular shift"""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift)


def amplitude_scale(x, low=0.8, high=1.2):
    """Random gain"""
    return x * np.random.uniform(low, high)


def time_warp(x, strength=0.1):
    """Non-linear time warping"""
    idx = np.arange(len(x))
    warp = idx + strength * np.sin(idx / 80)
    return np.interp(idx, warp, x)


# ============================================================
#  SIGNAL CLASS DEFINITIONS
# ============================================================

def class_0_normal():
    """Smooth healthy signal (sinusoidal + minor noise)"""
    t = np.linspace(0, 8*np.pi, SAMPLES)
    x = 0.7 * np.sin(t) + 0.2 * np.sin(3*t)
    return x


def class_1_breathing():
    """Low-frequency oscillations (breathing-like)"""
    t = np.linspace(0, 4*np.pi, SAMPLES)
    x = 1.0 * np.sin(t) + 0.1 * np.sin(6*t)
    return x


def class_2_burst():
    """Burst signals (short spikes + oscillations)"""
    t = np.linspace(0, 10*np.pi, SAMPLES)
    x = 0.5 * np.sin(5*t)

    # Add bursts
    for _ in range(8):
        pos = np.random.randint(0, SAMPLES-200)
        x[pos:pos+200] += np.hamming(200) * np.random.uniform(1, 2)

    return x


def class_3_mechanical():
    """Mechanical anomaly: irregular vibration + random impacts"""
    t = np.linspace(0, 20*np.pi, SAMPLES)
    x = 0.3 * np.sin(15*t) + 0.1 * np.sin(2*t)

    # Random mechanical shocks
    for _ in range(12):
        pos = np.random.randint(0, SAMPLES-100)
        x[pos:pos+100] += np.random.uniform(0.8, 1.5) * np.hanning(100)

    return x


def class_4_stress():
    """Strong distorted waveform + amplitude instability"""
    t = np.linspace(0, 12*np.pi, SAMPLES)
    x = 1.2 * np.sin(t * (1 + 0.1*np.sin(t/40)))   # nonlinear distortions
    x += 0.4 * np.sin(8*t)

    # Sudden spikes
    spikes = np.random.randint(0, SAMPLES, 15)
    x[spikes] += np.random.uniform(2, 5)

    return x


# Mapping
CLASS_GENERATORS = {
    0: class_0_normal,
    1: class_1_breathing,
    2: class_2_burst,
    3: class_3_mechanical,
    4: class_4_stress
}

# ============================================================
#  FULL REALISTIC GENERATION PIPELINE
# ============================================================


def generate_signal(label):
    """Generate one realistic, noisy, shifted, warped signal"""

    x = CLASS_GENERATORS[label]()  # base clean class pattern

    # Apply realistic imperfections
    x = add_noise(x)
    x = add_drift(x)
    x = random_shift(x)
    x = amplitude_scale(x)
    x = time_warp(x)

    return x


def generate_dataset(num_per_class=100):
    """Generate full dataset (X, y)"""
    X = []
    y = []

    for label in range(5):
        for _ in range(num_per_class):
            X.append(generate_signal(label))
            y.append(label)

    # Shuffle
    X = np.array(X)
    y = np.array(y)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ============================================================
#  RUN GENERATOR
# ============================================================

X, y = generate_dataset(num_per_class=100)

print("Generated dataset:")
print("X shape:", X.shape)   # (500, 12800)
print("y shape:", y.shape)

np.save("signals.npy", X)
np.save("labels.npy", y)

print("Saved signals.npy and labels.npy")
