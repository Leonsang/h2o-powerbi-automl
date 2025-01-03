# 📋 FAQ (Preguntas Frecuentes)

## ❓ Preguntas Comunes

### 1. Instalación y Configuración

**P: ¿Cómo instalo el sistema en Windows?**
```bash
# 1. Clonar repositorio
git clone https://github.com/Leonsang/h2o-powerbi-automl.git
cd h2o-powerbi-automl

# 2. Ejecutar instalador
install.bat
```

**P: ¿Qué requisitos de sistema necesito?**
- Python 3.8+
- Java 8+
- 8GB RAM mínimo
- Power BI Desktop

### 2. Uso Básico

**P: ¿Cómo empiezo con un modelo simple?**
```python
from src.IntegradorH2O_PBI import H2OModeloAvanzado

# Crear modelo
modelo = H2OModeloAvanzado()

# Entrenar
resultado = modelo.entrenar(datos)
```

**P: ¿Cómo interpreto los resultados?**
```python
# Ver métricas
print(resultado['metricas'])

# Explicación
explicacion = modelo.explicar_prediccion(id=123)
```

## 🔧 Solución de Problemas

### 1. Errores Comunes

**P: Error "H2O no inicia"**
- Verificar Java instalado
- Comprobar puertos libres
- Revisar memoria disponible

**P: Problemas de memoria**
```python
# Configurar memoria
modelo = H2OModeloAvanzado(memoria_maxima='4g')
```

### 2. Optimización

**P: ¿Cómo mejoro el rendimiento?**
```python
# Configuración optimizada
config = {
    'nthreads': -1,
    'max_models': 10,
    'early_stopping': True
}
modelo.entrenar(datos, **config)
```

## 💡 Tips y Trucos

### 1. Mejores Prácticas

**P: ¿Cómo preparo mejor mis datos?**
```python
# Preparación recomendada
datos_prep = modelo.preparar_datos(
    datos,
    limpiar=True,
    normalizar=True,
    eliminar_outliers=True
)
```

### 2. Características Avanzadas

**P: ¿Cómo uso ensembles?**
```python
# Configurar ensemble
ensemble = modelo.crear_ensemble(
    modelos=['gbm', 'rf', 'xgboost'],
    metalearner='glm'
)
```

## 📚 Recursos Adicionales

### 1. Enlaces Útiles
- [Documentación H2O](https://docs.h2o.ai/)
- [Power BI Docs](https://docs.microsoft.com/power-bi/)
- [Python API Reference](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/)

### 2. Soporte
- GitHub Issues
- Comunidad H2O
- Foro Power BI 