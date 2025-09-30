import numpy as np
import matplotlib.pyplot as plt

# Fijamos semilla para reproducibilidad
np.random.seed(42)

# Generamos datos sintéticos
n_samples = 200

# Lager (clase 0): menor IBU y RMS
lager_ibu = np.random.normal(loc=15, scale=5, size=n_samples//2)
lager_rms = np.random.normal(loc=20, scale=5, size=n_samples//2)
lager = np.column_stack((lager_ibu, lager_rms))
lager_y = np.zeros(n_samples//2)

# Stout (clase 1): mayor IBU y RMS
stout_ibu = np.random.normal(loc=40, scale=6, size=n_samples//2)
stout_rms = np.random.normal(loc=60, scale=6, size=n_samples//2)
stout = np.column_stack((stout_ibu, stout_rms))
stout_y = np.ones(n_samples//2)

# Concatenamos dataset completo
X = np.vstack((lager, stout))
y = np.concatenate((lager_y, stout_y))

# Mezclamos aleatoriamente
indices = np.arange(n_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split 80% train, 20% validation
split = int(0.8 * n_samples)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

