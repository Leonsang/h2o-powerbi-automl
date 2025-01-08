# Primeros Pasos

## Visión General
Esta guía te ayudará a empezar con el sistema H2O AutoML + IA Interpretable, desde la carga de datos hasta la interpretación de resultados.

## Ejemplo Básico

### 1. Importar Dependencias
```python
from src.IntegradorH2O_PBI import H2OModeloAvanzado
from src.asistente_ia import AsistenteDataScience
import pandas as pd

# Inicializar componentes
modelo = H2OModeloAvanzado()
asistente = AsistenteDataScience()
```

### 2. Cargar y Preparar Datos
```python
# Cargar datos
datos = pd.read_csv('datos/ventas.csv')

# Verificar estructura
print(f"Dimensiones: {datos.shape}")
print("\nColumnas:")
for col in datos.columns:
    print(f"- {col}: {datos[col].dtype}")

# Preparación básica
datos = modelo.preparar_datos(
    datos,
    objetivo='ventas',
    categoricas=['region', 'producto'],
    fecha='fecha_venta'
)
```

### 3. Entrenar Modelo
```python
# Configuración personalizada
config = {
    'tiempo_maximo': 600,    # 10 minutos
    'max_modelos': 20,       # Máximo 20 modelos
    'validacion_cruzada': True,
    'metricas': ['RMSE', 'MAE', 'R2']
}

# Entrenar modelo
resultado = modelo.entrenar(
    datos=datos,
    objetivo='ventas',
    config=config
)

print("\nMejor modelo:", resultado['tipo_modelo'])
print("Métricas:", resultado['metricas'])
```

### 4. Analizar Resultados
```python
# Generar análisis con IA
analisis = asistente.interpretar_resultados(resultado)
print("\nAnálisis IA:")
print(analisis)

# Visualizar importancia de variables
modelo.plot_importancia_variables(resultado['importancia_variables'])

# Generar explicaciones SHAP
modelo.plot_shap_summary(resultado['shap_values'])
```

## Uso en Power BI

### 1. Script Básico
```python
# En Power BI > Obtener Datos > Script Python
from src.script_pbi import ejecutar_prediccion

# El dataset es proporcionado automáticamente por Power BI
resultado = ejecutar_prediccion(dataset)
```

### 2. Visualizaciones Recomendadas
```python
# Crear visuales automáticos
from src.visualizaciones import crear_visuales_powerbi

visuales = crear_visuales_powerbi(
    datos=dataset,
    predicciones=resultado['predicciones'],
    metricas=resultado['metricas']
)
```

## Funcionalidades Avanzadas

### 1. Validación Cruzada
```python
# Evaluar modelo con validación cruzada
metricas_cv = modelo.validar_cruzado(
    datos=datos,
    objetivo='ventas',
    k_folds=5
)

print("\nMétricas por fold:")
for fold, metricas in metricas_cv.items():
    print(f"Fold {fold}:", metricas)
```

### 2. Explicaciones Locales
```python
# Explicar predicciones específicas
explicaciones = modelo.explicar_predicciones(
    datos=datos.iloc[0:5],
    modelo=resultado['modelo']
)

for i, exp in enumerate(explicaciones):
    print(f"\nCaso {i+1}:")
    print(exp)
```

### 3. Monitoreo de Rendimiento
```python
# Monitorear drift y rendimiento
from src.monitoreo import MonitorRendimiento

monitor = MonitorRendimiento()
reporte = monitor.evaluar_modelo(
    modelo=resultado['modelo'],
    datos_nuevos=datos_nuevos
)

if reporte['drift_detectado']:
    print("¡Alerta! Drift detectado")
    print("Recomendación:", reporte['recomendacion'])
```

## Mejores Prácticas

### 1. Preparación de Datos
- Verificar tipos de datos
- Tratar valores nulos
- Normalizar si es necesario
- Validar calidad de datos
- Documentar transformaciones

### 2. Entrenamiento
- Usar validación cruzada
- Monitorear tiempo de entrenamiento
- Guardar modelos importantes
- Documentar configuración
- Validar resultados

### 3. Interpretación
- Revisar métricas globales
- Analizar casos específicos
- Validar con expertos del dominio
- Documentar insights
- Mantener registro de decisiones

### 4. Mantenimiento
- Monitorear rendimiento
- Detectar drift de datos
- Reentrenar periódicamente
- Mantener logs
- Actualizar documentación

## Siguientes Pasos
1. [Flujo de Trabajo](04-flujo-trabajo.md)
2. [Análisis Avanzado](05-analisis.md)
3. [AutoML Avanzado](06-automl-avanzado.md)

## Requisitos Previos

### Versiones Soportadas
```bash
# Verificar versiones instaladas
python --version  # Debe ser 3.9-3.11
pip list | grep -E "h2o|pandas|numpy|scikit-learn"
```

### Entorno Virtual Recomendado
```bash
# Crear entorno con Python específico
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
``` 