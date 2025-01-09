# Mantenimiento y Monitoreo

## Visión General
Este módulo proporciona herramientas y prácticas para mantener y monitorear el sistema en producción.

## Sistema de Logging

### 1. Configuración de Logs
```python
from src.logger import Logger

# Inicializar logger
logger = Logger('mantenimiento')

# Configurar niveles
logger.set_level('INFO')  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### 2. Monitoreo de Eventos
```python
def monitorear_sistema():
    """Monitorea eventos del sistema"""
    try:
        # Verificar componentes
        logger.info("Iniciando verificación de sistema")
        verificar_h2o()
        verificar_modelos()
        verificar_cache()
        
    except Exception as e:
        logger.error(f"Error en monitoreo: {str(e)}")
        raise
```

## Gestión de Modelos

### 1. Versionado de Modelos
```python
from src.modelo_manager import ModeloManager

manager = ModeloManager()

# Guardar versión
manager.guardar_version(
    modelo=resultado['modelo'],
    metricas=resultado['metricas'],
    metadata={
        'version': '1.2.0',
        'fecha': '2024-02-20',
        'autor': 'data_scientist'
    }
)

# Cargar versión específica
modelo_anterior = manager.cargar_version('1.1.0')
```

### 2. Monitoreo de Drift
```python
from src.monitoreo import MonitorDrift

monitor = MonitorDrift()

# Detectar data drift
drift_report = monitor.detectar_drift(
    modelo=modelo_actual,
    datos_nuevos=datos_nuevos,
    umbral=0.05
)

if drift_report['drift_detectado']:
    logger.warning("Data drift detectado")
    notificar_equipo(drift_report)
```

## Mantenimiento Automático

### 1. Reentrenamiento Automático
```python
from src.mantenimiento import MantenimientoModelo

mantenimiento = MantenimientoModelo()

# Evaluar necesidad de reentrenamiento
if mantenimiento.evaluar_reentrenamiento(
    modelo=modelo_actual,
    datos_nuevos=datos_nuevos,
    metricas_objetivo=metricas_objetivo
):
    logger.info("Iniciando reentrenamiento automático")
    nuevo_modelo = mantenimiento.reentrenar(
        modelo_base=modelo_actual,
        datos_nuevos=datos_nuevos
    )
```

### 2. Limpieza de Recursos
```python
def limpiar_recursos():
    """Limpia recursos del sistema"""
    # Limpiar caché
    from src.cache import CacheManager
    cache = CacheManager()
    cache.limpiar_antiguos(dias=7)
    
    # Rotar logs
    logger.rotar_logs()
    
    # Liberar memoria H2O
    import h2o
    h2o.remove_all()
    h2o.cluster().show_status()
```

## Monitoreo de Rendimiento

### 1. Métricas del Sistema
```python
from src.monitoreo import MonitorRendimiento

monitor = MonitorRendimiento()

# Recolectar métricas
metricas = monitor.recolectar_metricas()
print("\nMétricas del sistema:")
for metrica, valor in metricas.items():
    print(f"- {metrica}: {valor}")
```

### 2. Alertas Automáticas
```python
from src.alertas import SistemaAlertas

alertas = SistemaAlertas()

# Configurar alertas
alertas.configurar({
    'memoria_maxima': '80%',
    'tiempo_respuesta': 5.0,
    'precision_minima': 0.8
})

# Monitoreo continuo
alertas.iniciar_monitoreo()
```

## Backups y Recuperación

### 1. Sistema de Backups
```python
from src.backup import BackupManager

backup = BackupManager()

# Backup completo
backup.crear_backup(
    modelos=True,
    configuracion=True,
    logs=True
)

# Restaurar desde backup
backup.restaurar('backup_2024_02_20.zip')
```

### 2. Plan de Recuperación
```python
def recuperacion_desastre():
    """Plan de recuperación ante fallos"""
    try:
        # 1. Detener servicios
        detener_servicios()
        
        # 2. Restaurar último backup
        backup.restaurar_ultimo()
        
        # 3. Verificar sistema
        verificar_sistema()
        
        # 4. Reiniciar servicios
        iniciar_servicios()
        
    except Exception as e:
        logger.critical(f"Error en recuperación: {str(e)}")
        notificar_emergencia()
```

## Mejores Prácticas

1. **Monitoreo Proactivo**
   - Revisar logs regularmente
   - Monitorear métricas clave
   - Configurar alertas

2. **Mantenimiento Regular**
   - Backups periódicos
   - Limpieza de recursos
   - Actualización de dependencias

3. **Documentación**
   - Mantener logs de cambios
   - Documentar incidentes
   - Actualizar procedimientos

## Siguientes Pasos
1. [Seguridad](10-seguridad.md)
2. [Optimización](11-optimizacion.md)
3. [FAQ](12-faq.md) 