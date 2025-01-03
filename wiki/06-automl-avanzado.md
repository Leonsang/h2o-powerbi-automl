# 🤖 AutoML Avanzado

## ⚙️ Configuración Avanzada

### Parámetros Principales
```python
configuracion = {
    'max_models': 20,          # Número máximo de modelos
    'max_runtime_secs': 300,   # Tiempo máximo de entrenamiento
    'seed': 1,                 # Semilla para reproducibilidad
    'nfolds': 5,              # Folds para validación cruzada
    'balance_classes': True,   # Balanceo de clases
    'include_algos': [         # Algoritmos a incluir
        'DRF', 'GLM', 'XGBoost', 'GBM', 'DeepLearning'
    ]
}
```

### Configuración por Algoritmo
```python
config_especifica = {
    'GBM': {
        'ntrees': 100,
        'max_depth': 5,
        'learn_rate': 0.1
    },
    'XGBoost': {
        'ntrees': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
}
```

## 🎯 Optimización de Hiperparámetros

### Grid Search
```python
# Definir grid
param_grid = {
    'ntrees': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learn_rate': [0.01, 0.1]
}

# Ejecutar búsqueda
mejor_modelo = modelo.grid_search(param_grid)
```

### Random Search
```python
# Configurar búsqueda aleatoria
random_params = {
    'ntrees': (50, 200),
    'max_depth': (3, 10),
    'learn_rate': (0.01, 0.1)
}

# Ejecutar búsqueda
mejor_modelo = modelo.random_search(random_params)
```

## 🔄 Selección de Modelos

### Leaderboard
```python
# Ver ranking de modelos
leaderboard = modelo.get_leaderboard()
print(leaderboard.head())
```

### Stacked Ensembles
```python
# Configurar ensemble
ensemble_config = {
    'base_models': ['GBM', 'RF', 'XGBoost'],
    'metalearner': 'GLM'
}

# Crear ensemble
modelo_ensemble = modelo.crear_ensemble(**ensemble_config)
```

## 📊 Ensambles

### Tipos de Ensambles
1. **Stacked Ensemble**
   - Mejor de familia
   - Todos los modelos
   - Mejores N modelos

2. **Super Learner**
   - Cross-validación
   - Metalearner
   - Pesos optimizados

### Ejemplo de Implementación
```python
# Crear super learner
super_learner = modelo.crear_super_learner(
    base_models=['GBM', 'RF', 'DL'],
    metalearner='GLM',
    folds=5
)

# Entrenar
resultado = super_learner.train(
    training_frame=datos_train,
    validation_frame=datos_valid
)
``` 