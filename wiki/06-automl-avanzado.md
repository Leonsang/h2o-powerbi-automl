# AutoML Avanzado

## Visión General
Este módulo proporciona funcionalidades avanzadas de AutoML con optimización automática e interpretabilidad integrada.

## Configuración Avanzada

### 1. Configuración de H2O
```python
from src.IntegradorH2O_PBI import H2OModeloAvanzado

# Configuración personalizada
config = {
    'max_models': 20,              # Número máximo de modelos
    'max_runtime_secs': 3600,      # Tiempo máximo de entrenamiento
    'stopping_rounds': 10,         # Early stopping
    'stopping_tolerance': 0.001,   # Tolerancia para early stopping
    'balance_classes': True,       # Balanceo de clases
    'seed': 42                     # Semilla para reproducibilidad
}

# Inicializar modelo
modelo = H2OModeloAvanzado(config=config)
```

### 2. Pipeline Automático
```python
# Configurar pipeline completo
pipeline_config = {
    'preprocesamiento': {
        'imputacion': 'auto',
        'encoding': 'auto',
        'scaling': True,
        'feature_selection': True
    },
    'modelado': {
        'algoritmos': ['GBM', 'RF', 'XGBoost', 'DeepLearning'],
        'validacion_cruzada': True,
        'n_folds': 5
    },
    'optimizacion': {
        'hyperopt': True,
        'n_trials': 100,
        'metric': 'auto'
    }
}

resultado = modelo.entrenar_avanzado(
    datos=datos,
    objetivo='target',
    config=pipeline_config
)
```

## Optimización Automática

### 1. Selección de Features
```python
from src.feature_selection import SelectorFeatures

# Selección automática
selector = SelectorFeatures()
features_importantes = selector.seleccionar_features(
    datos=datos,
    objetivo='target',
    metodo='auto'
)

print("\nFeatures seleccionadas:")
print(features_importantes)
```

### 2. Optimización de Hiperparámetros
```python
from src.optimizacion import OptimizadorHiperparametros

# Optimización bayesiana
optimizador = OptimizadorHiperparametros()
mejores_params = optimizador.optimizar(
    modelo=resultado['modelo'],
    datos=datos,
    objetivo='target',
    n_trials=100
)

print("\nMejores hiperparámetros:")
print(mejores_params)
```

## Ensamble y Stacking

### 1. Ensamble Automático
```python
from src.ensamble import EnsambleManager

# Crear ensamble
ensamble = EnsambleManager()
modelo_ensamble = ensamble.crear_ensamble(
    modelos=resultado['modelos'],
    metodo='weighted',
    weights='auto'
)

# Evaluar ensamble
metricas_ensamble = ensamble.evaluar(datos_test)
print("\nMétricas del ensamble:")
print(metricas_ensamble)
```

### 2. Stacking Avanzado
```python
# Configurar stacking
stacking_config = {
    'base_models': ['GBM', 'RF', 'XGBoost'],
    'meta_model': 'GLM',
    'folds': 5,
    'use_predictions': True
}

modelo_stacking = ensamble.crear_stacking(
    config=stacking_config,
    datos=datos
)
```

## Validación Avanzada

### 1. Validación Cruzada Estratificada
```python
from src.validacion import ValidadorAvanzado

validador = ValidadorAvanzado()
resultados_cv = validador.validacion_cruzada(
    modelo=resultado['modelo'],
    datos=datos,
    objetivo='target',
    n_folds=5,
    stratify=True
)

print("\nResultados validación cruzada:")
print(resultados_cv)
```

### 2. Validación Temporal
```python
# Validación con series temporales
resultados_temporales = validador.validacion_temporal(
    modelo=resultado['modelo'],
    datos=datos,
    fecha='fecha',
    ventana_tiempo='1M',
    n_splits=3
)
```

## Monitoreo y Mantenimiento

### 1. Monitoreo de Drift
```python
from src.monitoreo import MonitorDrift

monitor = MonitorDrift()
drift_report = monitor.detectar_drift(
    modelo=resultado['modelo'],
    datos_nuevos=datos_nuevos,
    umbral=0.05
)

print("\nReporte de drift:")
print(drift_report)
```

### 2. Reentrenamiento Automático
```python
from src.mantenimiento import MantenimientoModelo

mantenimiento = MantenimientoModelo()
necesita_reentrenar = mantenimiento.evaluar_reentrenamiento(
    modelo=resultado['modelo'],
    datos_nuevos=datos_nuevos,
    metricas_objetivo=metricas_objetivo
)

if necesita_reentrenar:
    nuevo_modelo = mantenimiento.reentrenar(
        modelo_base=resultado['modelo'],
        datos_nuevos=datos_nuevos
    )
```

## Mejores Prácticas

1. **Optimización**
   - Usar validación cruzada
   - Balancear tiempo/rendimiento
   - Monitorear recursos

2. **Validación**
   - Validar en múltiples escenarios
   - Usar datos representativos
   - Documentar resultados

3. **Mantenimiento**
   - Monitorear drift regularmente
   - Planificar reentrenamientos
   - Mantener versiones

## Siguientes Pasos
1. [Power BI](07-powerbi.md)
2. [Explicabilidad](08-explicabilidad.md)
3. [Mantenimiento](09-mantenimiento.md)

## Compatibilidad

### Versiones Soportadas
- H2O: 3.46.0.1
- Python: 3.9-3.11
- Pandas: 2.0.3
- NumPy: 1.24.3
- Scikit-learn: 1.3.0

### Limitaciones Conocidas
- No compatible con Python 3.12+
- Requiere Java 8+
- RAM mínima: 16GB 