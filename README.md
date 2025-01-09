# H2O AutoML Integration for Power BI

Sistema avanzado de AutoML con H2O.ai integrado con Power BI, que incluye monitoreo, MLOps y explicabilidad.

[![CI/CD](https://github.com/usuario/h2o-powerbi-automl/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/usuario/h2o-powerbi-automl/actions/workflows/ci_cd.yml)
[![codecov](https://codecov.io/gh/usuario/h2o-powerbi-automl/branch/main/graph/badge.svg)](https://codecov.io/gh/usuario/h2o-powerbi-automl)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Características Principales

- **AutoML con H2O.ai**: Entrenamiento automatizado de modelos
- **Feature Engineering Avanzado**: 
  - Detección y tratamiento de outliers
  - Transformaciones automáticas
  - Selección de features
  - Generación de interacciones
- **Monitoreo y MLOps**:
  - Detección de drift
  - Monitoreo de performance
  - Gestión de experimentos
  - Versionamiento de modelos
  - Despliegue automatizado
- **Explicabilidad**:
  - SHAP values
  - LIME explanations
  - Análisis global y local
  - Counterfactuals
- **Integración Power BI**:
  - Visualizaciones interactivas
  - Análisis en tiempo real
  - Dashboards automatizados

## Estructura del Proyecto

```
src/
├── features/                # Feature engineering
│   ├── feature_engineering.py
│   └── transformers.py
├── monitoring/             # Monitoreo y drift detection
│   ├── model_monitor.py
│   └── drift_detector.py
├── mlops/                  # MLOps y gestión de modelos
│   ├── mlops_manager.py
│   ├── experiment_tracker.py
│   └── model_registry.py
├── interpretability/       # Explicabilidad
│   ├── explainer.py
│   └── visualizations.py
└── powerbi/               # Integración Power BI
    ├── connector.py
    └── dashboard.py
```

## Instalación

```bash
pip install h2o-powerbi-automl
```

## Uso Rápido

```python
from h2o_powerbi_automl import AutoMLManager, FeatureEngineer, ModelMonitor

# Inicializar componentes
automl = AutoMLManager()
engineer = FeatureEngineer()
monitor = ModelMonitor()

# Preparar datos
features = engineer.process_features(data)

# Entrenar modelo
model = automl.train(features, target)

# Monitorear y explicar
monitor.check_model_health(model, new_data)
explanations = model.explain_predictions(new_data)
```

## Feature Engineering

El sistema incluye feature engineering avanzado:

```python
from h2o_powerbi_automl.features import FeatureEngineer

engineer = FeatureEngineer()

# Proceso completo
processed_data = engineer.process_features(
    data,
    outlier_method='isolation_forest',
    transform_method='robust',
    feature_selection='mutual_info'
)
```

## Monitoreo y MLOps

Sistema completo de monitoreo y MLOps:

```python
from h2o_powerbi_automl.mlops import MLOpsManager

mlops = MLOpsManager()

# Registrar experimento y modelo
mlops.track_and_register_model(
    model_info=model,
    metrics=metrics,
    parameters=params
)

# Desplegar y monitorear
mlops.deploy_and_monitor(
    model_name='my_model',
    environment='production'
)
```

## Explicabilidad

Herramientas avanzadas de explicabilidad:

```python
from h2o_powerbi_automl.interpretability import ModelExplainer

explainer = ModelExplainer(model)

# Explicación global
global_explanation = explainer.explain_global()

# Explicación local
local_explanation = explainer.explain_instance(instance)
```

## CI/CD y Desarrollo

El proyecto utiliza GitHub Actions para CI/CD:

1. **Tests Automáticos**:
   ```bash
   pytest tests/
   ```

2. **Linting y Formato**:
   ```bash
   pylint src/ tests/
   black src/ tests/
   ```

3. **Coverage**:
   ```bash
   pytest --cov=src tests/
   ```

## Contribución

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles.