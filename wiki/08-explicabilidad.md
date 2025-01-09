# Explicabilidad e Interpretabilidad

## Visión General
Este módulo proporciona herramientas avanzadas para interpretar y explicar los modelos usando una combinación de técnicas tradicionales y asistencia de IA.

## Componentes de Explicabilidad

### 1. SHAP (SHapley Additive exPlanations)
```python
from src.interpretabilidad import Interpretador
from src.visualizaciones import Visualizador

# Calcular SHAP values
interpretador = Interpretador()
shap_values = interpretador.calcular_shap_values(
    modelo=resultado['modelo'],
    datos=datos
)

# Visualizar
viz = Visualizador()
viz.plot_shap_summary(shap_values)
viz.plot_shap_dependence(shap_values, feature='precio')
```

### 2. LIME (Local Interpretable Model-agnostic Explanations)
```python
# Generar explicaciones locales
explicaciones_lime = interpretador.generar_lime_explicaciones(
    modelo=resultado['modelo'],
    datos=datos,
    num_features=5
)

# Visualizar explicación específica
for explicacion in explicaciones_lime:
    viz.plot_lime_explanation(explicacion)
```

### 3. Counterfactuals con DiCE
```python
# Generar counterfactuals
counterfactuals = interpretador.generar_counterfactuals(
    modelo=resultado['modelo'],
    datos=datos
)

# Analizar cambios necesarios
for cf in counterfactuals:
    print(f"\nCaso {cf['caso']}:")
    print("Cambios necesarios:")
    for cambio in cf['cambios']:
        print(f"- {cambio}")
```

## Asistente IA para Interpretación

### 1. Análisis Global
```python
from src.asistente_ia import AsistenteDataScience

asistente = AsistenteDataScience()

# Interpretar resultados globales
interpretacion = asistente.interpretar_resultados({
    'metricas': resultado['metricas'],
    'importancia_variables': resultado['importancia_variables'],
    'shap_values': shap_values
})

print("\nInterpretación IA:")
print(interpretacion)
```

### 2. Explicaciones Locales
```python
# Explicar predicciones específicas
for caso in resultado['casos_especiales']:
    explicacion = asistente.explicar_predicciones(
        caso=caso,
        explicacion_lime=explicaciones_lime[caso['id']],
        counterfactual=counterfactuals[caso['id']]
    )
    print(f"\nExplicación caso {caso['id']}:")
    print(explicacion)
```

### 3. Recomendaciones Técnicas
```python
# Generar recomendaciones
recomendaciones = asistente.generar_recomendaciones_tecnicas({
    'modelo': resultado['modelo'],
    'metricas': resultado['metricas'],
    'shap_values': shap_values
})

print("\nRecomendaciones técnicas:")
print(recomendaciones)
```

## Visualizaciones Avanzadas

### 1. Dashboards Interactivos
```python
from src.visualizaciones import DashboardExplicabilidad

dashboard = DashboardExplicabilidad()
dashboard.generar_dashboard(
    resultado=resultado,
    shap_values=shap_values,
    lime_explicaciones=explicaciones_lime,
    counterfactuals=counterfactuals
)
```

### 2. Reportes Automáticos
```python
from src.reportes import GeneradorReportes

generador = GeneradorReportes()
reporte = generador.generar_reporte_explicabilidad(
    resultado=resultado,
    interpretacion_ia=interpretacion,
    formato='html'
)
```

## Integración con Power BI

### 1. Visuales Personalizados
```python
def crear_visuales_explicabilidad(datos):
    """Crea visuales para Power BI"""
    return {
        'shap_summary': viz.plot_shap_summary(...),
        'lime_explicaciones': viz.plot_lime_explicaciones(...),
        'counterfactuals': viz.plot_counterfactuals(...)
    }
```

### 2. Insights Automáticos
```python
def generar_insights_powerbi(resultado):
    """Genera insights para Power BI"""
    insights = asistente.generar_insights({
        'shap': resultado['shap_values'],
        'lime': resultado['lime_explicaciones'],
        'counterfactuals': resultado['counterfactuals']
    })
    return pd.DataFrame(insights)
```

## Mejores Prácticas

1. **Interpretabilidad Global**
   - Usar múltiples técnicas
   - Validar con expertos
   - Documentar hallazgos

2. **Explicaciones Locales**
   - Personalizar por audiencia
   - Validar con casos conocidos
   - Mantener consistencia

3. **Comunicación**
   - Adaptar nivel técnico
   - Usar visualizaciones claras
   - Proporcionar contexto

## Siguientes Pasos
1. [Mantenimiento](09-mantenimiento.md)
2. [Seguridad](10-seguridad.md)
3. [Optimización](11-optimizacion.md) 