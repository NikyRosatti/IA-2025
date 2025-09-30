# IA-03 ML-Caso de Estudio - Python

## Introducción
- Importancia de los datos: más y mejores datos suelen superar modelos complejos.
- Ejemplo clásico: error de muestreo en encuesta de Literary Digest (1936).
- **Teorema No Free Lunch**: no existe un modelo que sea el mejor para todos los problemas.

## Proceso de Modelado
1. Definición del problema
2. Recolección de datos
3. Limpieza y preprocesamiento
4. Análisis exploratorio
5. Selección/ingeniería de características
6. Selección de modelo
7. Entrenamiento
8. Evaluación y ajuste
9. Implementación

## Caso de Estudio: California Housing Prices
- Predicción del valor medio de casas en distritos de California.
- Dataset con 10 características (longitud, latitud, edad de viviendas, habitaciones, ingresos, etc.).
- Problema de **regresión múltiple**.

## Métricas
- **RMSE** (Root Mean Squared Error): sensible a outliers.
- **MAE** (Mean Absolute Error): menos sensible a outliers.

## Recolección y Preparación de Datos
- Dataset descargado desde GitHub.
- Limpieza: tratamiento de valores faltantes (ej. `total_bedrooms`).
- Codificación de variables categóricas (`ocean_proximity`): OneHotEncoder u OrdinalEncoder.
- Escalado de características: Min-Max Scaling o estandarización.

## Exploración de Datos
- Histogramas, dispersión geográfica, correlaciones (Pearson).
- Nuevas características: rooms_per_house, bedrooms_ratio, people_per_house.

## División de Datos
- Train/Test Split (80/20).
- Muestreo estratificado según categoría de ingresos (`income_cat`).

## Implementación en Python
- Librerías: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
- Uso de `SimpleImputer`, `train_test_split`, `OneHotEncoder`.
- Ejemplo de pipeline para preprocesamiento y modelado.

