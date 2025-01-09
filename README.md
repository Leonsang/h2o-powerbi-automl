# H2O AutoML Integration for Power BI

Este proyecto proporciona una integración avanzada entre H2O AutoML y Power BI, permitiendo el entrenamiento automatizado de modelos de machine learning y su análisis detallado.

## Características Principales

- **Integración H2O AutoML**: Entrenamiento automatizado de modelos usando H2O.ai
- **Análisis Automático**: Generación de métricas, visualizaciones y análisis detallados
- **Interpretabilidad**: Explicaciones detalladas de las predicciones y comportamiento del modelo
- **Gestión de Modelos**: Sistema completo de gestión del ciclo de vida de modelos
- **Logging Avanzado**: Sistema robusto de logging con monitoreo y alertas

## Estructura del Proyecto

```
src/
├── __init__.py
├── analisis_manager.py      # Gestión de análisis de modelos
├── analisis_modelo.py       # Análisis detallado de modelos
├── analizar_resultados.py   # Análisis de resultados y métricas
├── asistente_ia.py         # Asistente IA para interpretación
├── config/
│   └── logging_config.json  # Configuración centralizada de logs
├── init_h2o_server.py      # Inicialización del servidor H2O
├── IntegradorH2O_PBI.py    # Integración principal con Power BI
├── interpretabilidad.py     # Herramientas de interpretabilidad
├── logger.py               # Sistema de logging
├── metricas.py            # Cálculo y gestión de métricas
├── modelo_manager.py       # Gestión de modelos
├── modelo_manager_ia.py    # Gestión de modelos con IA
├── script_pbi.py          # Script principal para Power BI
├── verificar_java.py      # Verificación de requisitos
└── visualizaciones.py     # Generación de visualizaciones
```

## Requisitos

- Python 3.8+
- H2O.ai
- Java 8+ (requerido por H2O)
- Bibliotecas Python (ver requirements.txt)

## Instalación

1. Clonar el repositorio
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Verificar instalación de Java:
   ```python
   from src.verificar_java import verificar_requisitos
   verificar_requisitos()
   ```

## Uso Básico

```python
from src import H2OModeloAvanzado, ejecutar_prediccion
import pandas as pd

# Cargar datos
datos = pd.read_csv('datos.csv')

# Ejecutar predicción con análisis completo
resultado = ejecutar_prediccion(
    datos=datos,
    tipo_modelo='automl',
    analisis_completo=True
)
```

## Sistema de Logging

El proyecto incluye un sistema avanzado de logging con las siguientes características:

- Rotación automática de archivos de log
- Niveles configurables por módulo
- Monitoreo y alertas
- Estadísticas de errores
- Formato enriquecido con contexto

### Configuración de Logs

La configuración se realiza mediante el archivo `config/logging_config.json`:

```json
{
    "log_dir": "logs",
    "default_level": "INFO",
    "handlers": {
        "file": {
            "enabled": true,
            "level": "DEBUG"
        },
        "console": {
            "enabled": true,
            "level": "INFO"
        }
    }
}
```

## Análisis de Modelos

El sistema proporciona análisis detallado de modelos incluyendo:

- Métricas de rendimiento
- Importancia de variables
- Análisis de errores
- Segmentación
- Tendencias y patrones
- Visualizaciones

## Interpretabilidad

Se incluyen herramientas para interpretación de modelos:

- SHAP values
- LIME explanations
- Análisis de importancia global
- Explicaciones locales
- Counterfactuals

## Contribución

1. Fork del repositorio
2. Crear rama para feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit de cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## Licencia

Este proyecto está licenciado bajo MIT License - ver archivo LICENSE para detalles.