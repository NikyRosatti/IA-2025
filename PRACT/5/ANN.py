import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from scipy import ndimage

# ============================================
# PARTE 1: CARGAR DATOS
# ============================================
def load_data():
    with gzip.open("mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data

def prepare_data_sklearn():
    """Prepara los datos en formato sklearn (sin vectorización one-hot)"""
    tr_d, va_d, te_d = load_data()
    
    X_train = tr_d[0]  # (50000, 784)
    y_train = tr_d[1]  # (50000,)
    
    X_val = va_d[0]    # (10000, 784)
    y_val = va_d[1]    # (10000,)
    
    X_test = te_d[0]   # (10000, 784)
    y_test = te_d[1]   # (10000,)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================
# PARTE 2a: MODELO BASE - COMPARACIÓN CON EJ 1
# ============================================
def train_baseline_model(X_train, y_train, X_test, y_test):
    """Modelo con arquitectura similar al ejercicio 1: [784, 30, 10]"""
    print("="*60)
    print("PARTE 2a: MODELO BASE (comparación con ejercicio 1)")
    print("="*60)
    
    # MLPClassifier con arquitectura similar: 1 capa oculta de 30 neuronas
    # Nota: sklearn maneja el learning rate diferente que la implementación manual
    model = MLPClassifier(
        hidden_layer_sizes=(30,),      # 1 capa oculta con 30 neuronas
        activation='logistic',          # sigmoid (como en el ej 1)
        solver='sgd',                   # SGD (como en el ej 1)
        learning_rate_init=0.5,         # Ajustado para sklearn (3.0 es demasiado alto)
        batch_size=10,                  # mini-batch de 10 (como en el ej 1)
        max_iter=30,                    # 30 épocas (como en el ej 1)
        momentum=0.0,                   # Sin momentum para ser más similar al ej 1
        random_state=42,
        verbose=True,
        learning_rate='constant',       # tasa de aprendizaje constante
        alpha=0.0001                    # Regularización por defecto
    )
    
    model.fit(X_train, y_train)
    
    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPrecisión en test: {accuracy:.4f} ({int(accuracy*len(y_test))}/{len(y_test)})")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Modelo Base')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.savefig('confusion_matrix_base.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, accuracy

# ============================================
# PARTE 2b: VARIACIÓN DE HIPERPARÁMETROS
# ============================================
def train_with_different_hyperparameters(X_train, y_train, X_test, y_test):
    """Entrena modelos con diferentes hiperparámetros"""
    print("\n" + "="*60)
    print("PARTE 2b: VARIACIÓN DE HIPERPARÁMETROS")
    print("="*60)
    
    results = {}
    
    # Modelo 1: Mayor regularización (alpha)
    print("\n1. Modelo con mayor regularización (L2)")
    model_reg = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='logistic',
        solver='sgd',
        learning_rate_init=0.5,
        batch_size=10,
        max_iter=30,
        momentum=0.0,
        alpha=0.1,  # Regularización L2 fuerte
        random_state=42,
        verbose=False
    )
    model_reg.fit(X_train, y_train)
    acc_reg = accuracy_score(y_test, model_reg.predict(X_test))
    results['Regularización L2 (α=0.1)'] = acc_reg
    print(f"Precisión: {acc_reg:.4f}")
    
    # Modelo 2: Optimizador Adam
    print("\n2. Modelo con optimizador Adam")
    model_adam = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='logistic',
        solver='adam',  # Optimizador Adam
        learning_rate_init=0.001,  # Adam usa tasa más baja
        batch_size=10,
        max_iter=30,
        random_state=42,
        verbose=False
    )
    model_adam.fit(X_train, y_train)
    acc_adam = accuracy_score(y_test, model_adam.predict(X_test))
    results['Optimizador Adam'] = acc_adam
    print(f"Precisión: {acc_adam:.4f}")
    
    # Modelo 3: Red más profunda
    print("\n3. Modelo con arquitectura más profunda [100, 50]")
    model_deep = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 capas ocultas
        activation='relu',  # ReLU es mejor para redes profundas
        solver='adam',
        learning_rate_init=0.001,
        batch_size=10,
        max_iter=30,
        random_state=42,
        verbose=False
    )
    model_deep.fit(X_train, y_train)
    acc_deep = accuracy_score(y_test, model_deep.predict(X_test))
    results['Red profunda [100, 50] + ReLU'] = acc_deep
    print(f"Precisión: {acc_deep:.4f}")
    
    # Modelo 4: Learning rate adaptativo
    print("\n4. Modelo con learning rate adaptativo")
    model_adaptive = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='logistic',
        solver='sgd',
        learning_rate='adaptive',  # Tasa adaptativa
        learning_rate_init=0.5,
        momentum=0.0,
        batch_size=10,
        max_iter=30,
        random_state=42,
        verbose=False
    )
    model_adaptive.fit(X_train, y_train)
    acc_adaptive = accuracy_score(y_test, model_adaptive.predict(X_test))
    results['Learning rate adaptativo'] = acc_adaptive
    print(f"Precisión: {acc_adaptive:.4f}")
    
    # Modelo 5: Sin regularización
    print("\n5. Modelo sin regularización (α=0)")
    model_no_reg = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='logistic',
        solver='sgd',
        learning_rate_init=0.5,
        momentum=0.0,
        batch_size=10,
        max_iter=30,
        alpha=0.0,  # Sin regularización
        random_state=42,
        verbose=False
    )
    model_no_reg.fit(X_train, y_train)
    acc_no_reg = accuracy_score(y_test, model_no_reg.predict(X_test))
    results['Sin regularización (α=0)'] = acc_no_reg
    print(f"Precisión: {acc_no_reg:.4f}")
    
    # Modelo 6: Learning rate más alto (optimizado)
    print("\n6. Modelo con learning rate alto optimizado")
    model_high_lr = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='logistic',
        solver='sgd',
        learning_rate_init=1.0,  # Learning rate más alto
        momentum=0.0,
        batch_size=10,
        max_iter=30,
        alpha=0.0001,
        random_state=42,
        verbose=False
    )
    model_high_lr.fit(X_train, y_train)
    acc_high_lr = accuracy_score(y_test, model_high_lr.predict(X_test))
    results['Learning rate alto (η=1.0)'] = acc_high_lr
    print(f"Precisión: {acc_high_lr:.4f}")
    
    # Visualizar comparación
    print("\n" + "-"*60)
    print("RESUMEN DE RESULTADOS:")
    print("-"*60)
    for name, acc in results.items():
        print(f"{name:35s}: {acc:.4f}")
    
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = plt.bar(range(len(names)), accuracies, color=colors)
    plt.xlabel('Configuración')
    plt.ylabel('Precisión')
    plt.title('Comparación de Precisión según Hiperparámetros')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylim([0.9, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores sobre las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{accuracies[i]:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model_deep  # Retornar el mejor modelo

# ============================================
# PARTE 2c: IMÁGENES TRANSFORMADAS
# ============================================
def create_transformed_images(X_test, y_test, n_samples=1000):
    """Crea imágenes con números más pequeños y desplazados"""
    print("\n" + "="*60)
    print("PARTE 2c: EVALUACIÓN CON IMÁGENES TRANSFORMADAS")
    print("="*60)
    
    # Seleccionar muestra aleatoria
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    # Transformación 1: Escalar (hacer más pequeño)
    X_scaled = []
    for img in X_sample:
        img_2d = img.reshape(28, 28)
        # Zoom de 0.7 (hace la imagen más pequeña)
        img_scaled = ndimage.zoom(img_2d, 0.7)
        # Pad para volver a 28x28
        pad_size = (28 - img_scaled.shape[0]) // 2
        img_padded = np.pad(img_scaled, pad_size, mode='constant', constant_values=0)
        if img_padded.shape[0] < 28:
            img_padded = np.pad(img_padded, ((0, 1), (0, 1)), mode='constant')
        X_scaled.append(img_padded[:28, :28].flatten())
    X_scaled = np.array(X_scaled)
    
    # Transformación 2: Desplazar a la esquina superior izquierda
    X_top_left = []
    for img in X_sample:
        img_2d = img.reshape(28, 28)
        img_shifted = ndimage.shift(img_2d, shift=(-7, -7), mode='constant', cval=0)
        X_top_left.append(img_shifted.flatten())
    X_top_left = np.array(X_top_left)
    
    # Transformación 3: Desplazar a la esquina inferior derecha
    X_bottom_right = []
    for img in X_sample:
        img_2d = img.reshape(28, 28)
        img_shifted = ndimage.shift(img_2d, shift=(7, 7), mode='constant', cval=0)
        X_bottom_right.append(img_shifted.flatten())
    X_bottom_right = np.array(X_bottom_right)
    
    # Transformación 4: Más pequeño + desplazado
    X_small_shifted = []
    for img in X_sample:
        img_2d = img.reshape(28, 28)
        img_scaled = ndimage.zoom(img_2d, 0.6)
        pad_size = (28 - img_scaled.shape[0]) // 2
        img_padded = np.pad(img_scaled, pad_size, mode='constant', constant_values=0)
        if img_padded.shape[0] < 28:
            img_padded = np.pad(img_padded, ((0, 1), (0, 1)), mode='constant')
        img_shifted = ndimage.shift(img_padded[:28, :28], shift=(5, 5), mode='constant', cval=0)
        X_small_shifted.append(img_shifted.flatten())
    X_small_shifted = np.array(X_small_shifted)
    
    return {
        'original': (X_sample, y_sample),
        'escalado_70%': (X_scaled, y_sample),
        'desplazado_sup_izq': (X_top_left, y_sample),
        'desplazado_inf_der': (X_bottom_right, y_sample),
        'pequeño_desplazado': (X_small_shifted, y_sample)
    }

def visualize_transformations(transformed_data):
    """Visualiza ejemplos de las transformaciones"""
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    
    transformations = list(transformed_data.keys())
    
    for i, trans_name in enumerate(transformations):
        X_trans, y_trans = transformed_data[trans_name]
        for j in range(5):
            idx = j * 20  # Espaciado para ver diferentes dígitos
            ax = axes[i, j]
            ax.imshow(X_trans[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(trans_name, rotation=0, ha='right', va='center', fontsize=9)
            ax.set_title(f'{y_trans[idx]}', fontsize=10)
    
    plt.suptitle('Ejemplos de Transformaciones Aplicadas', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('transformations_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Ejemplos visuales guardados en 'transformations_examples.png'")

def evaluate_on_transformations(models_dict, transformed_data):
    """Evalúa todos los modelos en las imágenes transformadas"""
    print("\n" + "-"*60)
    print("EVALUACIÓN EN IMÁGENES TRANSFORMADAS:")
    print("-"*60)
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{model_name}:")
        results[model_name] = {}
        
        for trans_name, (X_trans, y_trans) in transformed_data.items():
            y_pred = model.predict(X_trans)
            acc = accuracy_score(y_trans, y_pred)
            results[model_name][trans_name] = acc
            print(f"  {trans_name:25s}: {acc:.4f}")
    
    # Visualizar resultados
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(transformed_data))
    width = 0.15
    multiplier = 0
    
    for model_name, trans_results in results.items():
        accuracies = list(trans_results.values())
        offset = width * multiplier
        bars = ax.bar(x + offset, accuracies, width, label=model_name)
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=7, rotation=90)
        
        multiplier += 1
    
    ax.set_ylabel('Precisión')
    ax.set_xlabel('Tipo de Transformación')
    ax.set_title('Robustez de los Modelos ante Transformaciones')
    ax.set_xticks(x + width * (len(models_dict) - 1) / 2)
    ax.set_xticklabels(list(transformed_data.keys()), rotation=45, ha='right')
    ax.legend(loc='lower left')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Gráfico de robustez guardado en 'robustness_comparison.png'")

# ============================================
# MAIN: EJECUTAR TODO
# ============================================
if __name__ == "__main__":
    print("Cargando datos MNIST...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_sklearn()
    print(f"Datos cargados: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Parte 2a: Modelo baseline
    model_base, acc_base = train_baseline_model(X_train, y_train, X_test, y_test)
    
    # Parte 2b: Diferentes hiperparámetros
    model_best = train_with_different_hyperparameters(X_train, y_train, X_test, y_test)
    
    # Parte 2c: Imágenes transformadas
    transformed_data = create_transformed_images(X_test, y_test, n_samples=1000)
    visualize_transformations(transformed_data)
    
    # Evaluar todos los modelos
    models_dict = {
        'Modelo Base (ej 1)': model_base,
        'Mejor Modelo': model_best
    }
    
    evaluate_on_transformations(models_dict, transformed_data)
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print("\nArchivos generados:")
    print("  - confusion_matrix_base.png")
    print("  - hyperparameter_comparison.png")
    print("  - transformations_examples.png")
    print("  - robustness_comparison.png")