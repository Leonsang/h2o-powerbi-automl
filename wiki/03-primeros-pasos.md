# 🎯 Primeros Pasos

## 📊 Preparación de Datos

### Formato Requerido
```python
import pandas as pd

# Cargar datos
datos = pd.read_csv('datos/ventas.csv')

# Estructura recomendada
datos.head()
"""
   fecha       producto  ventas  precio  region
0  2023-01-01  A        100     15.99   Norte
1  2023-01-01  B        75      24.99   Sur
"""
```

### Limpieza Básica
```python
from src.IntegradorH2O_PBI import H2OModeloAvanzado

# Inicializar modelo
modelo = H2OModeloAvanzado()

# Preparar datos
datos_limpios = modelo.preparar_datos(
    datos,
    objetivo='ventas',
    categoricas=['producto', 'region']
)
```

## 🤖 Entrenamiento Básico

### Ejemplo Simple
```python
# Entrenar modelo
resultado = modelo.entrenar(datos_limpios)

# Ver métricas básicas
print(resultado['metricas'])
```

### Validación Rápida
```python
# Validar modelo
metricas = modelo.validar(datos_test)
print(f"R²: {metricas['r2']:.3f}")
```

## 📈 Visualización de Resultados

### En Python
```python
# Graficar predicciones vs reales
modelo.plot_predicciones()

# Importancia de variables
modelo.plot_importancia_variables()
```

### En Power BI
1. Cargar script Python
2. Usar visual personalizado
3. Configurar parámetros

## 📊 Interpretación de Métricas

### Métricas Clave
- R² (Coeficiente de determinación)
- RMSE (Error cuadrático medio)
- MAE (Error absoluto medio)

### Ejemplo de Interpretación
```python
# Obtener explicación
explicacion = modelo.explicar_prediccion(
    prediccion_id=123,
    nivel_detalle='detallado'
)
print(explicacion['descripcion'])
``` 