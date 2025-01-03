# 📊 Análisis y Métricas

## 📈 Métricas Disponibles

### Métricas de Regresión
```python
metricas = {
    'r2': 'Coeficiente de determinación',
    'rmse': 'Error cuadrático medio',
    'mae': 'Error absoluto medio',
    'mape': 'Error porcentual absoluto medio'
}
```

### Métricas de Clasificación
```python
metricas = {
    'accuracy': 'Precisión general',
    'precision': 'Precisión positiva',
    'recall': 'Sensibilidad',
    'f1': 'Media armónica P/R',
    'auc': 'Área bajo la curva ROC'
}
```

## 🔍 Interpretación de Resultados

### Análisis de Predicciones
```python
from src.analizar_resultados import analizar_resultados

# Análisis completo
analisis = analizar_resultados(
    modelo=modelo,
    datos=datos_test,
    predicciones=predicciones
)

# Interpretación
print(analisis['interpretacion'])
```

### Importancia de Variables
```python
# Top variables
importancia = modelo.obtener_importancia_variables()
print(importancia.head())
```

## 📊 Visualizaciones

### Gráficos Básicos
```python
# Predicciones vs Reales
modelo.plot_predicciones()

# Residuos
modelo.plot_residuos()

# Importancia de Variables
modelo.plot_importancia_variables()
```

### Gráficos Avanzados
```python
# Análisis de Componentes
modelo.plot_pca()

# Matriz de Correlación
modelo.plot_correlacion()

# SHAP Values
modelo.plot_shap_values()
```

## 💡 Mejores Prácticas

### 1. Validación de Modelos
- Usar validación cruzada
- Separar datos de test
- Validar en diferentes períodos

### 2. Interpretación
- Revisar todas las métricas
- Analizar casos extremos
- Validar con expertos

### 3. Monitoreo
- Seguimiento de drift
- Alertas de degradación
- Actualización periódica 