import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_fscore_support)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CARGAR Y PREPARAR DATOS
# ============================================
print("="*70)
print("CARGANDO DATASET IRIS")
print("="*70)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"\nDimensiones del dataset: {X.shape}")
print(f"Clases: {target_names}")
print(f"Features: {feature_names}")
print(f"\nDistribución de clases:")
for i, name in enumerate(target_names):
    print(f"  {name}: {np.sum(y == i)} muestras")

# Dividir datos: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de prueba: {X_test.shape[0]} muestras")

# Normalizar datos (importante para MLP y SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 1. MULTI-LAYER PERCEPTRON (MLP)
# ============================================
print("\n" + "="*70)
print("1. ENTRENANDO MULTI-LAYER PERCEPTRON (MLP)")
print("="*70)

# MLP con diferentes configuraciones
mlp_configs = {
    'MLP Simple [10]': MLPClassifier(
        hidden_layer_sizes=(10,),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    ),
    'MLP Profunda [20, 10]': MLPClassifier(
        hidden_layer_sizes=(20, 10),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    ),
    'MLP con Regularización': MLPClassifier(
        hidden_layer_sizes=(15,),
        activation='relu',
        solver='adam',
        alpha=0.01,  # Regularización L2
        max_iter=1000,
        random_state=42
    ),
    'MLP Sigmoid': MLPClassifier(
        hidden_layer_sizes=(10,),
        activation='logistic',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
}

mlp_results = {}

for name, mlp in mlp_configs.items():
    print(f"\n{name}:")
    mlp.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = mlp.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5)
    
    mlp_results[name] = {
        'model': mlp,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"  Precisión en test: {accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Seleccionar mejor MLP
best_mlp_name = max(mlp_results, key=lambda k: mlp_results[k]['accuracy'])
best_mlp = mlp_results[best_mlp_name]['model']
print(f"\n⭐ Mejor MLP: {best_mlp_name}")

# ============================================
# 2. SUPPORT VECTOR MACHINE (SVM)
# ============================================
print("\n" + "="*70)
print("2. ENTRENANDO SUPPORT VECTOR MACHINE (SVM)")
print("="*70)

svm_configs = {
    'SVM Linear': SVC(kernel='linear', random_state=42),
    'SVM RBF': SVC(kernel='rbf', gamma='scale', random_state=42),
    'SVM Poly': SVC(kernel='poly', degree=3, random_state=42),
    'SVM RBF (C=10)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
}

svm_results = {}

for name, svm in svm_configs.items():
    print(f"\n{name}:")
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    
    svm_results[name] = {
        'model': svm,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"  Precisión en test: {accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

best_svm_name = max(svm_results, key=lambda k: svm_results[k]['accuracy'])
best_svm = svm_results[best_svm_name]['model']
print(f"\n⭐ Mejor SVM: {best_svm_name}")

# ============================================
# 3. DECISION TREE
# ============================================
print("\n" + "="*70)
print("3. ENTRENANDO DECISION TREE")
print("="*70)

tree_configs = {
    'Decision Tree (sin poda)': DecisionTreeClassifier(random_state=42),
    'Decision Tree (max_depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Decision Tree (max_depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Decision Tree (min_samples_split=10)': DecisionTreeClassifier(
        min_samples_split=10, random_state=42
    )
}

tree_results = {}

for name, tree in tree_configs.items():
    print(f"\n{name}:")
    # Decision Tree no requiere normalización, usar datos originales
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    cv_scores = cross_val_score(tree, X_train, y_train, cv=5)
    
    tree_results[name] = {
        'model': tree,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"  Precisión en test: {accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

best_tree_name = max(tree_results, key=lambda k: tree_results[k]['accuracy'])
best_tree = tree_results[best_tree_name]['model']
print(f"\n⭐ Mejor Decision Tree: {best_tree_name}")

# ============================================
# 4. COMPARACIÓN GENERAL
# ============================================
print("\n" + "="*70)
print("4. COMPARACIÓN DE TODOS LOS MODELOS")
print("="*70)

# Crear tabla comparativa
comparison_data = []

for name, result in mlp_results.items():
    comparison_data.append({
        'Modelo': name,
        'Tipo': 'MLP',
        'Precisión Test': result['accuracy'],
        'CV Mean': result['cv_mean'],
        'CV Std': result['cv_std']
    })

for name, result in svm_results.items():
    comparison_data.append({
        'Modelo': name,
        'Tipo': 'SVM',
        'Precisión Test': result['accuracy'],
        'CV Mean': result['cv_mean'],
        'CV Std': result['cv_std']
    })

for name, result in tree_results.items():
    comparison_data.append({
        'Modelo': name,
        'Tipo': 'Decision Tree',
        'Precisión Test': result['accuracy'],
        'CV Mean': result['cv_mean'],
        'CV Std': result['cv_std']
    })

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Precisión Test', ascending=False)

print("\n" + df_comparison.to_string(index=False))

# ============================================
# 5. VISUALIZACIONES
# ============================================
print("\n" + "="*70)
print("5. GENERANDO VISUALIZACIONES")
print("="*70)

# 5.1 Gráfico de barras comparativo
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Comparación general por tipo
ax1 = axes[0, 0]
df_grouped = df_comparison.groupby('Tipo')['Precisión Test'].mean()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(df_grouped.index, df_grouped.values, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Precisión Promedio', fontsize=12)
ax1.set_title('Precisión Promedio por Tipo de Modelo', fontsize=14, fontweight='bold')
ax1.set_ylim([0.85, 1.0])
ax1.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 2: Comparación de mejores modelos
ax2 = axes[0, 1]
best_models = {
    best_mlp_name: mlp_results[best_mlp_name]['accuracy'],
    best_svm_name: svm_results[best_svm_name]['accuracy'],
    best_tree_name: tree_results[best_tree_name]['accuracy']
}
models = list(best_models.keys())
accuracies = list(best_models.values())
colors_best = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax2.bar(range(len(models)), accuracies, color=colors_best, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
ax2.set_ylabel('Precisión', fontsize=12)
ax2.set_title('Comparación de Mejores Modelos', fontsize=14, fontweight='bold')
ax2.set_ylim([0.85, 1.0])
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 3: Cross-validation scores
ax3 = axes[1, 0]
cv_data = df_comparison[['Modelo', 'CV Mean', 'CV Std', 'Tipo']].copy()
cv_data = cv_data.sort_values('CV Mean', ascending=True)

y_pos = np.arange(len(cv_data))
colors_map = {'MLP': '#FF6B6B', 'SVM': '#4ECDC4', 'Decision Tree': '#45B7D1'}
colors_cv = [colors_map[tipo] for tipo in cv_data['Tipo']]

ax3.barh(y_pos, cv_data['CV Mean'], xerr=cv_data['CV Std'], 
         color=colors_cv, alpha=0.8, edgecolor='black', capsize=5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(cv_data['Modelo'], fontsize=8)
ax3.set_xlabel('CV Score', fontsize=12)
ax3.set_title('Cross-Validation Scores (5-Fold)', fontsize=14, fontweight='bold')
ax3.set_xlim([0.85, 1.0])
ax3.grid(axis='x', alpha=0.3)

# Subplot 4: Todos los modelos
ax4 = axes[1, 1]
df_sorted = df_comparison.sort_values('Precisión Test', ascending=True)
y_pos = np.arange(len(df_sorted))
colors_all = [colors_map[tipo] for tipo in df_sorted['Tipo']]

ax4.barh(y_pos, df_sorted['Precisión Test'], color=colors_all, alpha=0.8, edgecolor='black')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(df_sorted['Modelo'], fontsize=8)
ax4.set_xlabel('Precisión Test', fontsize=12)
ax4.set_title('Ranking de Todos los Modelos', fontsize=14, fontweight='bold')
ax4.set_xlim([0.85, 1.0])
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('iris_comparison_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico general guardado: iris_comparison_overview.png")

# 5.2 Matrices de confusión de los mejores modelos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

best_models_dict = {
    'MLP': (best_mlp, mlp_results[best_mlp_name]['predictions']),
    'SVM': (best_svm, svm_results[best_svm_name]['predictions']),
    'Decision Tree': (best_tree, tree_results[best_tree_name]['predictions'])
}

for idx, (model_type, (model, predictions)) in enumerate(best_models_dict.items()):
    cm = confusion_matrix(y_test, predictions)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[idx], cbar_kws={'label': 'Cuenta'})
    
    axes[idx].set_title(f'{model_type}\nPrecisión: {accuracy_score(y_test, predictions):.4f}',
                       fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Clase Real', fontsize=11)
    axes[idx].set_xlabel('Clase Predicha', fontsize=11)

plt.tight_layout()
plt.savefig('iris_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Matrices de confusión guardadas: iris_confusion_matrices.png")

# 5.3 Reportes de clasificación detallados
print("\n" + "="*70)
print("6. REPORTES DE CLASIFICACIÓN DETALLADOS")
print("="*70)

for model_type, (model, predictions) in best_models_dict.items():
    print(f"\n{model_type}:")
    print("-" * 70)
    print(classification_report(y_test, predictions, target_names=target_names))

# 5.4 Análisis de importancia de features (Decision Tree)
if hasattr(best_tree, 'feature_importances_'):
    print("\n" + "="*70)
    print("7. IMPORTANCIA DE FEATURES (Decision Tree)")
    print("="*70)
    
    importances = best_tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nRanking de features:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], 
            color='#45B7D1', alpha=0.8, edgecolor='black')
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importancia', fontsize=12)
    plt.title('Importancia de Features - Decision Tree', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('iris_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Gráfico de importancia guardado: iris_feature_importance.png")

# ============================================
# 8. RESUMEN FINAL
# ============================================
print("\n" + "="*70)
print("8. RESUMEN Y CONCLUSIONES")
print("="*70)

print("\n🏆 GANADOR GENERAL:")
overall_winner = df_comparison.iloc[0]
print(f"   Modelo: {overall_winner['Modelo']}")
print(f"   Tipo: {overall_winner['Tipo']}")
print(f"   Precisión: {overall_winner['Precisión Test']:.4f}")
print(f"   CV Score: {overall_winner['CV Mean']:.4f} (+/- {overall_winner['CV Std']:.4f})")

print("\n📊 ESTADÍSTICAS POR TIPO:")
for tipo in ['MLP', 'SVM', 'Decision Tree']:
    subset = df_comparison[df_comparison['Tipo'] == tipo]
    print(f"\n{tipo}:")
    print(f"  Mejor precisión: {subset['Precisión Test'].max():.4f}")
    print(f"  Precisión promedio: {subset['Precisión Test'].mean():.4f}")
    print(f"  Desviación estándar: {subset['Precisión Test'].std():.4f}")

print("\n💡 CONCLUSIONES:")
print("  • Iris es un dataset pequeño y linealmente separable")
print("  • Todos los modelos logran >90% de precisión")
print("  • MLP puede ser excesivo para este problema simple")
print("  • SVM y Decision Tree son más eficientes e interpretables")
print("  • La normalización beneficia a MLP y SVM pero no a Decision Tree")

print("\n📁 ARCHIVOS GENERADOS:")
print("  - iris_comparison_overview.png")
print("  - iris_confusion_matrices.png")
print("  - iris_feature_importance.png")

print("\n" + "="*70)
print("ANÁLISIS COMPLETADO")
print("="*70)