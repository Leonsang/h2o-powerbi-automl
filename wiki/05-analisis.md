# Análisis y Diagnóstico

## Visión General
Este módulo proporciona análisis profundo y diagnóstico de modelos usando interpretabilidad asistida por IA.

## Componentes de Análisis

### 1. Análisis Global
```python
from src.analisis_modelo import AnalizadorModelo
from src.asistente_ia import AsistenteDataScience

# Inicializar componentes
analizador = AnalizadorModelo(output_dir='resultados/')
asistente = AsistenteDataScience()

# Generar análisis completo
analisis = analizador.analisis_completo(
    modelo=resultado['modelo'],
    datos=datos,
    predicciones=resultado['predicciones'],
    objetivo='target',
    tipo_modelo='regresion'
)
```

### 2. Métricas y Diagnósticos
```python
from src.metricas import Metricas

# Calcular métricas
metricas = Metricas()
diagnostico = metricas.calcular_metricas_globales(
    reales=datos['target'],
    predicciones=resultado['predicciones'],
    tipo_modelo='regresion'
)

print("\nMétricas principales:")
for metrica, valor in diagnostico.items():
    print(f"- {metrica}: {valor}")
```

### 3. Visualizaciones Avanzadas
```python
from src.visualizaciones import Visualizador

viz = Visualizador()

# Gráficos de diagnóstico
viz.plot_predicciones_vs_reales(reales, predicciones)
viz.plot_distribucion_errores(reales, predicciones)
viz.plot_shap_summary(analisis['shap_values'])
viz.plot_importancia_variables(analisis['importancia_variables'])
```

## Interpretabilidad IA

### 1. Análisis Automático
```python
# Generar interpretación
interpretacion = asistente.interpretar_resultados({
    'metricas': diagnostico,
    'importancia_variables': analisis['importancia_variables'],
    'insights': analisis['insights']
})

print("\nInterpretación IA:")
print(interpretacion)
```

### 2. Explicaciones Locales
```python
# Explicar predicciones específicas
for caso in analisis['casos_especiales']:
    explicacion = asistente.explicar_predicciones(
        caso=caso['id'],
        explicacion_lime=caso['lime'],
        counterfactual=caso['counterfactual']
    )
    print(f"\nCaso {caso['id']}:")
    print(explicacion)
```

### 3. Recomendaciones Técnicas
```python
# Obtener recomendaciones
recomendaciones = asistente.generar_recomendaciones_tecnicas({
    'metricas': diagnostico,
    'shap_values': analisis['shap_values']
})

print("\nRecomendaciones técnicas:")
print(recomendaciones)
```

## Exportación de Resultados

### 1. Guardar Análisis
```python
from src.exportador import Exportador

exportador = Exportador(output_dir='resultados/')
exportador.guardar_analisis_completo(
    analisis=analisis,
    interpretacion=interpretacion,
    recomendaciones=recomendaciones
)
```

### 2. Generar Reporte
```python
# Generar reporte HTML
exportador.generar_reporte_html(
    titulo="Análisis Completo del Modelo",
    descripcion="Análisis detallado con interpretabilidad IA"
)

# Exportar para Power BI
exportador.exportar_powerbi(
    analisis=analisis,
    formato='pbix'
)
```

## Mejores Prácticas

1. **Análisis Sistemático**
   - Seguir protocolo de análisis
   - Documentar hallazgos
   - Validar interpretaciones

2. **Visualización**
   - Usar gráficos apropiados
   - Mantener consistencia
   - Facilitar comparaciones

3. **Interpretación**
   - Contrastar con expertos
   - Validar insights
   - Documentar decisiones

## Siguientes Pasos
1. [AutoML Avanzado](06-automl-avanzado.md)
2. [Power BI](07-powerbi.md)
3. [Explicabilidad](08-explicabilidad.md)

### Interpretabilidad
- SHAP (v0.41.0)
- LIME (v0.2.0.1)
- DiCE (v0.9)

### Visualización
- Matplotlib (v3.7.2)
- Seaborn (v0.12.2)
- Plotly (v5.18.0) 