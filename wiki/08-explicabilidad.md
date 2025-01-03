# 🔍 Explicabilidad

## 🎯 Interpretación de Modelos

### LIME (Local Interpretable Model-agnostic Explanations)
```python
from src.IntegradorH2O_PBI import H2OModeloAvanzado

# Explicar predicción específica
explicacion = modelo.explicar_prediccion(
    id_prediccion=123,
    metodo='lime',
    num_features=5
)
```

### Importancia Global
```python
# Obtener importancia global
importancia = modelo.obtener_importancia_global()

# Visualizar
modelo.plot_importancia_global()
```

## 🎨 SHAP Values

### Cálculo de SHAP
```python
# Calcular SHAP values
shap_values = modelo.calcular_shap_values(datos_test)

# Visualizar
modelo.plot_shap_summary()
```

### Interpretación Local
```python
# Explicar una instancia
explicacion_local = modelo.explicar_instancia(
    instancia_id=456,
    metodo='shap'
)
```

## 📊 Importancia de Variables

### Ranking de Variables
```python
# Obtener ranking
ranking = modelo.ranking_variables()
print(ranking.head())
```

### Dependencia Parcial
```python
# Gráficos de dependencia
modelo.plot_dependencia_parcial(
    variable='precio',
    interaccion='region'
)
```

## 🔄 Análisis de Predicciones

### Explicaciones en Lenguaje Natural
```python
# Generar explicación
explicacion = modelo.generar_explicacion_natural(
    prediccion_id=789,
    nivel_detalle='alto'
)
```

### Dashboard de Explicabilidad
```python
# Crear dashboard
dashboard = modelo.crear_dashboard_explicabilidad(
    predicciones=predicciones_recientes,
    metricas=['shap', 'importancia', 'residuos']
)
``` 