# 🔌 Integración Power BI

## 🔄 Conexión con Power BI

### 1. Configuración Inicial
```python
# En Power BI, usar Script Python
from src.script_pbi import conectar_h2o

# Configurar conexión
conexion = conectar_h2o(
    puerto=54321,
    memoria="4g"
)
```

### 2. Importar Datos
```python
# Cargar datos desde Power BI
datos = dataset
resultado = modelo.predecir(datos)
```

## 📜 Scripts Disponibles

### Predicción Automática
```python
# script_prediccion.py
from src.IntegradorH2O_PBI import H2OModeloAvanzado

modelo = H2OModeloAvanzado()
predicciones = modelo.predecir(dataset)
```

### Análisis de Resultados
```python
# script_analisis.py
from src.analizar_resultados import analizar_resultados

reporte = analizar_resultados(modelo, dataset, predicciones)
```

## 📊 Visualizaciones Personalizadas

### 1. Gráfico de Predicciones
```python
# visual_predicciones.py
import seaborn as sns

def plot_predicciones(reales, predichas):
    return sns.regplot(x=reales, y=predichas)
```

### 2. Importancia de Variables
```python
# visual_importancia.py
def plot_importancia(modelo):
    return modelo.plot_variable_importance()
```

## 🔄 Actualización de Datos

### Actualización Automática
1. Configurar Schedule
2. Definir triggers
3. Gestionar refrescos

### Monitoreo
- Logs de actualización
- Alertas de errores
- Métricas de rendimiento 