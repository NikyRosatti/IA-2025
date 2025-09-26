import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = "clasif_lineal_epochs"
os.makedirs(out_dir, exist_ok=True)

# Dataset (x0 = 1 implícito en el bias)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])  # OR lógico

# Configuración inicial
w0, w1, w2 = 0.0, 0.0, 0.0   # pesos iniciales (w0 = bias)
n_epochs = 4                 # cantidad de épocas
alpha = 1.0                  # tasa de aprendizaje

def predict(w0, w1, w2, x):
    h = w0 + w1*x[0] + w2*x[1]
    return 1 if h >= 0 else 0, h

# Guardar pesos por época
epoch_weights = []

for epoch in range(n_epochs):
    # Guardar pesos actuales
    epoch_weights.append((w0, w1, w2))
    
    for xi, yi in zip(X, y):
        y_pred, h = predict(w0, w1, w2, xi)
        error = yi - y_pred
        # Regla de actualización: wi ← wi + α (y - h(x)) xi
        w0 += alpha * error * 1      # x0 = 1
        w1 += alpha * error * xi[0]
        w2 += alpha * error * xi[1]

# Markdown y gráficas
md_lines = ["# Resultados por época\n",
            "Regla de actualización: `wi ← wi + α (y - h(x)) xi`\n\n",
            f"Dataset: `X = {X.tolist()}`, `y = {y.tolist()}`\n\n"]

fig_all, axes_all = plt.subplots(1, n_epochs, figsize=(4*n_epochs, 4))

for i, (w0, w1, w2) in enumerate(epoch_weights):
    h_vals = []
    preds = []
    for xi in X:
        pred, h = predict(w0, w1, w2, xi)
        h_vals.append(h)
        preds.append(pred)

    md_lines.append(f"## Época {i+1} — pesos (w0={w0:.2f}, w1={w1:.2f}, w2={w2:.2f})\n")
    md_lines.append("| Caso | x1 | x2 | h(x) | Predicción | Clase real | Correcto |\n")
    md_lines.append("|---:|:--:|:--:|:--:|:--:|:--:|:--:|\n")
    for j, xi in enumerate(X):
        case = f"[{int(xi[0])},{int(xi[1])}]"
        hx = h_vals[j]
        pred = preds[j]
        true = int(y[j])
        correct = "✅" if pred == true else "❌"
        md_lines.append(f"| {case} | {int(xi[0])} | {int(xi[1])} | {hx:.2f} | {pred} | {true} | {correct} |\n")
    md_lines.append("\n")

    # Gráfica
    ax_all = axes_all[i] if n_epochs > 1 else axes_all
    ax_all.set_xlim(-0.2, 1.2)
    ax_all.set_ylim(-0.2, 1.2)
    ax_all.grid(True, linestyle="--", alpha=0.4)
    ax_all.set_title(f"Época {i+1}")

    # Dibujar puntos
    for xi, true, pred in zip(X, y, preds):
        color = "red" if true == 0 else "blue"
        marker = "x" if pred == 1 else "o"
        ax_all.scatter(xi[0], xi[1], marker=marker, s=120,
                       edgecolor="k", facecolor=color, linewidth=0.7)

    # Dibujar línea de decisión
    if abs(w2) > 1e-8:
        x_vals = np.linspace(-0.5, 1.5, 200)
        ax_all.plot(x_vals, -(w0 + w1*x_vals)/w2, color="green", linewidth=2)
    elif abs(w1) > 1e-8:
        ax_all.axvline(x=-w0/w1, color="green", linewidth=2)

# Guardar figura
combined_path = os.path.join(out_dir, "epochs_grid.png")
fig_all.tight_layout()
fig_all.savefig(combined_path, bbox_inches="tight", dpi=150)
plt.close(fig_all)

# Guardar Markdown
with open(os.path.join(out_dir, "epochs_table.md"), "w", encoding="utf-8") as f:
    f.write("".join(md_lines))

print("Entrenamiento completo. Resultados guardados en:", out_dir)
