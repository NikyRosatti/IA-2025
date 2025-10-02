import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ----------------------------
# 1) Generar dataset sintético
# ----------------------------
# Generamos 300 muestras: 150 "no-stout" (Lager-like) y 150 "stout".
n_per_class = 150

# Lager: IBU ~ N(18, 4), RMS ~ N(24, 6)
lager_ibu = np.random.normal(loc=18, scale=4, size=n_per_class)
lager_rms = np.random.normal(loc=24, scale=6, size=n_per_class)

# Stout: IBU ~ N(46, 5), RMS ~ N(58, 6)
stout_ibu = np.random.normal(loc=46, scale=5, size=n_per_class)
stout_rms = np.random.normal(loc=58, scale=6, size=n_per_class)

X_lager = np.vstack([lager_ibu, lager_rms]).T
X_stout = np.vstack([stout_ibu, stout_rms]).T

X = np.vstack([X_lager, X_stout])
y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])  # 0 = not-stout, 1 = stout

# Añadimos un poco de mezcla (ruido) para no hacerlo trivial
noise = np.random.normal(scale=0.6, size=X.shape)
X += noise

# ----------------------------
# 2) Entrenamiento / Validación split
# ----------------------------
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

train_frac = 0.75
m_train = int(len(X) * train_frac)
X_train, X_val = X[:m_train], X[m_train:]
y_train, y_val = y[:m_train], y[m_train:]

print(f"Total: {len(X)}, Train: {len(X_train)}, Val: {len(X_val)}")

# ----------------------------
# 3) Funciones necesarias
# ----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost(X, Y, W, B):
    """
    Cost (log-loss) for logistic regression.
    X: (m,n), Y: (m,1), W: (n,1), B: scalar
    """
    m = X.shape[0]
    Z = np.dot(X, W) + B
    A = sigmoid(Z)
    eps = 1e-9
    cost = - (1/m) * np.sum(Y * np.log(A + eps) + (1 - Y) * np.log(1 - A + eps))
    return cost

def train_logistic_regression(X_train, y_train, X_val=None, y_val=None,
                              learning_rate=0.05, iterations=2000, print_every=200):
    m, n = X_train.shape
    Y_train = y_train.reshape(m, 1)
    W = np.zeros((n,1))
    B = 0.0

    train_costs = []
    val_costs = []

    for it in range(iterations+1):
        # Forward
        Z = np.dot(X_train, W) + B   # (m,1)
        A = sigmoid(Z)

        # Cost
        cost = compute_cost(X_train, Y_train, W, B)
        train_costs.append(cost)

        # Validation cost if provided
        if X_val is not None and y_val is not None:
            val_costs.append(compute_cost(X_val, y_val.reshape(-1,1), W, B))

        # Gradients
        dZ = A - Y_train              # (m,1)
        dW = (1/m) * np.dot(X_train.T, dZ)  # (n,1)
        dB = (1/m) * np.sum(dZ)

        # Update
        W -= learning_rate * dW
        B -= learning_rate * dB

        if it % print_every == 0:
            if X_val is not None and y_val is not None:
                print(f"Iter {it:5d}: train_cost = {cost:.4f}, val_cost = {val_costs[-1]:.4f}")
            else:
                print(f"Iter {it:5d}: train_cost = {cost:.4f}")

    return W, B, train_costs, val_costs

def predict_proba(X, W, B):
    Z = np.dot(X, W) + B
    return sigmoid(Z)  # shape (m,1)

def predict(X, W, B, threshold=0.5):
    probs = predict_proba(X, W, B)
    return (probs >= threshold).astype(int)

def metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, f1 (binary)
    y_true, y_pred: 1D arrays or column vectors with values 0/1.
    """
    y_t = y_true.flatten()
    y_p = y_pred.flatten()
    tp = np.sum((y_t==1) & (y_p==1))
    tn = np.sum((y_t==0) & (y_p==0))
    fp = np.sum((y_t==0) & (y_p==1))
    fn = np.sum((y_t==1) & (y_p==0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

# ----------------------------
# 4) Entrenar (manual) regresión logística
# ----------------------------
W, B, train_costs, val_costs = train_logistic_regression(
    X_train, y_train, X_val, y_val,
    learning_rate=0.05, iterations=2000, print_every=200
)

# ----------------------------
# 5) Evaluación: Train vs Val
# ----------------------------
train_preds = predict(X_train, W, B)
val_preds = predict(X_val, W, B)

m_train_metrics = metrics(y_train.reshape(-1,1), train_preds)
m_val_metrics = metrics(y_val.reshape(-1,1), val_preds)

print("\n--- Métricas en TRAIN ---")
for k,v in m_train_metrics.items():
    if k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:9s}: {v*100:.2f}%")
    else:
        print(f"{k:9s}: {v}")

print("\n--- Métricas en VAL ---")
for k,v in m_val_metrics.items():
    if k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:9s}: {v*100:.2f}%")
    else:
        print(f"{k:9s}: {v}")

# ----------------------------
# 6) Visualizaciones
# ----------------------------

# 6.1 Scatter (train + val), frontera de decisión y mapa de probabilidades
plt.figure(figsize=(8,6))
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], s=20, alpha=0.6, label="Train: not-stout", marker='o')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], s=20, alpha=0.6, label="Train: stout", marker='x')
plt.scatter(X_val[y_val==0,0], X_val[y_val==0,1], s=30, edgecolor='k', facecolor='none', linewidth=1.2, label="Val: not-stout (holdout)")
plt.scatter(X_val[y_val==1,0], X_val[y_val==1,1], s=30, edgecolor='r', facecolor='none', linewidth=1.2, label="Val: stout (holdout)")

# Grid for probability heatmap
ibu_min, ibu_max = X[:,0].min()-3, X[:,0].max()+3
rms_min, rms_max = X[:,1].min()-3, X[:,1].max()+3
xx, yy = np.meshgrid(np.linspace(ibu_min, ibu_max, 200), np.linspace(rms_min, rms_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs_grid = predict_proba(grid, W, B).reshape(xx.shape)

plt.contourf(xx, yy, probs_grid, levels=20, alpha=0.25)  # probability heatmap (soft)
# Decision boundary (prob=0.5) contour
cs = plt.contour(xx, yy, probs_grid, levels=[0.5], colors='green', linewidths=2)
cs.collections[0].set_label("Decision boundary (p=0.5)")

plt.xlabel("IBU")
plt.ylabel("RMS")
plt.legend(loc='upper left')
plt.title("Datos (train/val), heatmap de probabilidad y frontera de decisión")
plt.show()

# 6.2 Curvas de costo (train vs val)
plt.figure(figsize=(7,4))
iters = np.arange(len(train_costs))
plt.plot(iters, train_costs, label="train cost")
plt.plot(iters[:len(val_costs)], val_costs, label="val cost")
plt.xlabel("Iteración")
plt.ylabel("Log-loss")
plt.legend()
plt.title("Evolución del costo (train vs val)")
plt.grid(True)
plt.show()

# 6.3 Sigmoide (visual)
x_vals = np.linspace(-10, 10, 300)
plt.figure(figsize=(6,3))
plt.plot(x_vals, sigmoid(x_vals))
plt.title("Función Sigmoide σ(x)")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.show()

# 6.4 Mostrar algunos ejemplos y sus probabilidades (val)
probs_val = predict_proba(X_val, W, B).flatten()
print("\nEjemplos de la partición de validación (primeros 10):")
for i in range(min(10, len(X_val))):
    print(f"X={X_val[i]}  true={int(y_val[i])}  p(stout)={probs_val[i]:.3f}  pred={int(val_preds[i])}")
