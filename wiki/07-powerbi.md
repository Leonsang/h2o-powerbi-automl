# Integración con Power BI

## Visión General
Este módulo proporciona una integración seamless entre H2O AutoML y Power BI, permitiendo análisis predictivos avanzados directamente en el entorno de BI.

## Configuración Inicial

### 1. Preparación del Entorno
```python
# powerbi_config.py
CONFIGURACION = {
    'python_path': '.venv/Scripts/python.exe',  # Windows
    # 'python_path': '.venv/bin/python',        # Linux/Mac
    'timeout': 300,
    'memoria_maxima': '8G',
    'logging': True
}
```

### 2. Script Principal
```python
# powerbi_script.py
from src.IntegradorH2O_PBI import H2OModeloAvanzado
from src.asistente_ia import AsistenteDataScience

def main(dataset):
    """Punto de entrada para Power BI"""
    try:
        # 1. Inicializar componentes
        modelo = H2OModeloAvanzado()
        asistente = AsistenteDataScience()
        
        # 2. Entrenar y analizar
        resultado = modelo.entrenar(
            datos=dataset,
            objetivo='target'
        )
        
        # 3. Generar insights
        analisis = asistente.interpretar_resultados(resultado)
        
        # 4. Preparar salida
        return {
            'predicciones': resultado['predicciones'],
            'metricas': resultado['metricas'],
            'analisis': analisis
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
```

## Uso en Power BI

### 1. Importar Datos
1. Obtener Datos → Script de Python
2. Pegar código base:
```python
from powerbi_script import main
resultado = main(dataset)
```

### 2. Transformaciones Automáticas
```python
# Preprocesamiento específico para Power BI
def preparar_datos_powerbi(datos):
    """Adapta datos para Power BI"""
    return {
        'datos_modelo': datos['predicciones'],
        'metricas_resumen': pd.DataFrame(datos['metricas']),
        'insights': pd.DataFrame(datos['analisis'])
    }
```

### 3. Visualizaciones Personalizadas
```python
# Generar visualizaciones
def crear_visualizaciones(datos):
    """Crea visualizaciones para Power BI"""
    from src.visualizaciones import Visualizador
    
    viz = Visualizador()
    graficos = {
        'predicciones': viz.plot_predicciones_vs_reales(...),
        'importancia': viz.plot_importancia_variables(...),
        'shap': viz.plot_shap_summary(...)
    }
    return graficos
```

## Funcionalidades Avanzadas

### 1. Actualización Incremental
```python
def actualizar_modelo(datos_nuevos):
    """Actualiza modelo con nuevos datos"""
    from src.mantenimiento import MantenimientoModelo
    
    mantenimiento = MantenimientoModelo()
    if mantenimiento.evaluar_reentrenamiento(...):
        return mantenimiento.reentrenar(...)
    return None
```

### 2. Caché Inteligente
```python
from src.cache import CacheManager

cache = CacheManager()
resultado = cache.get_or_compute(
    key='modelo_actual',
    compute_fn=lambda: modelo.entrenar(...),
    ttl='1d'
)
```

### 3. Procesamiento por Lotes
```python
def procesar_lote(datos, batch_size=1000):
    """Procesa datos en lotes"""
    resultados = []
    for i in range(0, len(datos), batch_size):
        lote = datos[i:i+batch_size]
        resultado = modelo.predecir(lote)
        resultados.append(resultado)
    return pd.concat(resultados)
```

## Optimización de Rendimiento

### 1. Gestión de Memoria
```python
def optimizar_memoria():
    """Optimiza uso de memoria"""
    import gc
    gc.collect()
    h2o.cluster().show_status()
    h2o.remove_all()
```

### 2. Paralelización
```python
from concurrent.futures import ThreadPoolExecutor

def procesar_paralelo(datos, n_workers=4):
    """Procesa datos en paralelo"""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(procesar_lote, lote) 
                  for lote in np.array_split(datos, n_workers)]
        return pd.concat([f.result() for f in futures])
```

## Mejores Prácticas

1. **Rendimiento**
   - Usar caché cuando sea posible
   - Procesar en lotes grandes
   - Optimizar memoria

2. **Mantenimiento**
   - Monitorear uso de recursos
   - Actualizar modelos regularmente
   - Mantener logs detallados

3. **Seguridad**
   - Validar inputs
   - Manejar errores gracefully
   - Proteger datos sensibles

## Troubleshooting

### Problemas Comunes
1. **Timeout en Power BI**
   - Aumentar timeout en configuración
   - Reducir tamaño de datos
   - Optimizar procesamiento

2. **Errores de Memoria**
   - Usar procesamiento por lotes
   - Limpiar cache regularmente
   - Ajustar configuración H2O

3. **Fallos de Conexión**
   - Verificar rutas Python
   - Comprobar dependencias
   - Revisar logs

## Siguientes Pasos
1. [Explicabilidad](08-explicabilidad.md)
2. [Mantenimiento](09-mantenimiento.md)
3. [Seguridad](10-seguridad.md) 