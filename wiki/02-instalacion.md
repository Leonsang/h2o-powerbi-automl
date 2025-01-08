# Guía de Instalación

## Requisitos del Sistema

### Hardware Recomendado
- CPU: 4+ cores
- RAM: 16GB mínimo (32GB recomendado para datasets grandes)
- Disco: 10GB espacio libre
- GPU: Opcional, mejora rendimiento de modelos deep learning

### Software Necesario
- Python 3.9-3.11 (3.9 recomendado)
- Java 8+ (requerido por H2O)
- Git
- Power BI Desktop (opcional)

### Versiones Específicas
```bash
# Versiones principales
python==3.9.x
h2o==3.46.0.1
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# Ver requirements.txt para lista completa
```

## Proceso de Instalación

### 1. Preparación del Entorno

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/h2o-automl-ia.git
cd h2o-automl-ia

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración del Sistema

#### Variables de Entorno
```bash
# Copiar template
cp .env.example .env

# Editar configuración
H2O_MEMORY=8G
H2O_PORT=54321
MODELO_IA=orca-mini
LOG_LEVEL=INFO
```

#### Verificación de Componentes
```python
# Ejecutar diagnóstico
python scripts/diagnostico.py

# Verificar H2O
python -c "import h2o; h2o.init()"

# Verificar modelo IA
python -c "from src.asistente_ia import AsistenteDataScience; AsistenteDataScience()"
```

### 3. Integración con Power BI

#### Configuración Python en Power BI
1. Abrir Power BI Desktop
2. Archivo > Opciones > Python Scripting
3. Configurar:
   - Directorio Python: ruta al entorno virtual
   - Timeout: 300 segundos mínimo

#### Script de Conexión
```python
# En Power BI > Obtener Datos > Script Python
from src.script_pbi import ejecutar_prediccion
resultado = ejecutar_prediccion(dataset)
```

## Estructura de Directorios

```
proyecto/
├── modelos/          # Modelos entrenados
├── logs/            # Logs del sistema
├── config/          # Configuraciones
├── src/             # Código fuente
├── tests/           # Tests
├── docs/            # Documentación
├── datos/           # Datos ejemplo
└── temp/           # Archivos temporales
```

## Verificación de Instalación

### 1. Tests Automáticos
```bash
# Ejecutar suite completa
python -m pytest tests/

# Tests específicos
python -m pytest tests/test_instalacion.py
python -m pytest tests/test_integracion.py
python -m pytest tests/test_pbi.py
```

### 2. Logs y Diagnóstico
```bash
# Ver logs
tail -f logs/h2o_server.log
tail -f logs/asistente_ia.log

# Diagnóstico completo
python scripts/diagnostico.py
```

## Troubleshooting

### Problemas Comunes

1. **Error de Memoria H2O**
   - Síntoma: H2O no inicia o se cierra inesperadamente
   - Solución: Ajustar H2O_MEMORY en .env
   - Verificar memoria disponible

2. **Fallo en Descarga de Modelos**
   - Síntoma: Error en verificar_modelo()
   - Solución: Verificar conexión/proxy
   - Intentar descarga manual

3. **Error Power BI**
   - Síntoma: No conecta con Python
   - Solución: Verificar ruta en configuración
   - Aumentar timeout

### Logs y Diagnóstico

```bash
# Ver logs detallados
tail -f logs/h2o_server.log
tail -f logs/asistente_ia.log

# Diagnóstico completo
python scripts/diagnostico.py
```

## Siguientes Pasos
1. [Primeros Pasos](03-primeros-pasos.md)
2. [Flujo de Trabajo](04-flujo-trabajo.md)
3. [Análisis](05-analisis.md) 