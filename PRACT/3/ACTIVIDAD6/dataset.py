import numpy as np
import matplotlib.pyplot as plt

# Fijamos semilla para reproducibilidad
np.random.seed(42)

# Generamos datos sintéticos
n_samples = 200

# Lager (clase 0): menor IBU y RMS
lager_ibu = np.random.normal(loc=15, scale=5, size=n_samples)
lager_rms = np.random.normal(loc=20, scale=5, size=n_samples)
lager = np.column_stack((lager_ibu, lager_rms))
lager_y = np.zeros(n_samples//2)

# Stout (clase 1): mayor IBU y RMS
stout_ibu = np.random.normal(loc=40, scale=6, size=n_samples)
stout_rms = np.random.normal(loc=60, scale=6, size=n_samples)
stout = np.column_stack((stout_ibu, stout_rms))
stout_y = np.ones(n_samples//2)

# Clase 2: IPA (amarga y no tan oscura Alto IBU, Bajo RMS)
ipa_ibu = np.random.normal(loc=50, scale=6, size=n_samples)
ipa_rms = np.random.normal(loc=25, scale=5, size=n_samples)
ipa_X = np.column_stack((ipa_ibu, ipa_rms))
ipa_y = np.full(n_samples//2, 2)

# Clase 3: Scottish (ligeramente oscura y no tan amarga Bajo IBU, Medio RMS)
scottish_ibu = np.random.normal(loc=20, scale=5, size=n_samples)
scottish_rms = np.random.normal(loc=40, scale=6, size=n_samples)
scottish_X = np.column_stack((scottish_ibu, scottish_rms))
scottish_y = np.full(n_samples//2, 3)

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

