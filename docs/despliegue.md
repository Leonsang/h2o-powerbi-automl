# Guía de Despliegue

## Requisitos del Sistema
- Python 3.8+
- H2O 3.46.0+
- Memoria RAM: 16GB recomendado
- CPU: 4+ cores
- Espacio en disco: 10GB mínimo
- GPU: Opcional, mejora rendimiento

## Instalación

1. **Entorno Virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. **Dependencias**
```bash
pip install -r requirements.txt
```

3. **Configuración**
- Copiar `.env.example` a `.env`
- Ajustar variables de entorno:
  * H2O_CONFIG
  * MODELO_IA
  * PATHS_CONFIG

## Componentes IA

1. **Modelos de Interpretabilidad**
- Los modelos se descargan automáticamente
- Verificación de integridad automática
- Caché inteligente de modelos

2. **Configuración de IA**
```python
from src.asistente_ia import AsistenteDataScience
asistente = AsistenteDataScience(modelo_base='orca-mini')
```

## Integración con Power BI

1. **Configuración Python**
- Verificar ruta Python en Power BI
- Instalar dependencias globalmente
- Configurar memoria y timeout

2. **Script de Conexión**
- Copiar `powerbi_script.py`
- Ajustar configuración según necesidades
- Verificar permisos de escritura

3. **Verificación**
```bash
python -m pytest tests/
```

## Monitoreo

1. **Logs**
- Sistema de logging multinivel
- Rotación automática de logs
- Monitoreo de recursos

2. **Métricas**
- Dashboard en tiempo real
- Alertas configurables
- Métricas de rendimiento

3. **Diagnósticos**
- Verificación de modelos
- Tests automáticos
- Validación de resultados

## Mantenimiento

1. **Backups**
- Modelos: Diario
- Configuración: Por cambio
- Logs: Semanal
- Cache: Mensual

2. **Actualizaciones**
```bash
# Actualizar dependencias
pip install --upgrade -r requirements.txt

# Actualizar modelos IA
python -m src.modelo_manager_ia --update-all

# Verificar sistema
python -m pytest tests/
```

3. **Limpieza**
```bash
# Limpiar cache
python scripts/clean_cache.py

# Rotar logs
python scripts/rotate_logs.py
```

## Troubleshooting

1. **Problemas Comunes**
- Errores de memoria
- Timeouts en Power BI
- Fallos de modelo

2. **Soluciones**
- Ajustar configuración de memoria
- Verificar logs específicos
- Reinstalar componentes

3. **Soporte**
- Documentación: /docs
- Logs: /logs
- Issues: GitHub 