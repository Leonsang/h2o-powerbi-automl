# 🚀 Guía de Despliegue H2O AutoML para Power BI

## 1. Preparación del Entorno

### 1.1 Requisitos Previos
```bash
# Verificar versiones
python --version  # Python 3.8+
java -version    # Java 8+
```

### 1.2 Clonar Repositorio
```bash
git clone https://github.com/Leonsang/h2o-powerbi-automl.git
cd h2o-powerbi-automl
```

## 2. Instalación

### 2.1 Windows
```bash
# Ejecutar instalador
install.bat
```

### 2.2 Linux/Mac
```bash
# Dar permisos y ejecutar
chmod +x install.sh
./install.sh
```

## 3. Verificación de Instalación

### 3.1 Estructura de Directorios
```bash
# Verificar directorios creados
tree
```

Deberías ver:
```
proyecto/
├── modelos/
├── logs/
│   ├── h2o/
│   ├── modelos/
│   └── tests/
├── config/
├── src/
├── tests/
├── docs/
├── datos/
└── temp/
    └── h2o_temp/
```

### 3.2 Tests
```bash
# Ejecutar suite completa de tests
python -m tests.test_suite
```

### 3.3 Verificar Logs
```bash
# Verificar creación de logs
ls logs/h2o/
ls logs/modelos/
ls logs/tests/
```

## 4. Configuración Power BI

### 4.1 Configurar Python en Power BI
1. Abrir Power BI Desktop
2. Ir a Archivo > Opciones > Python Scripting
3. Configurar ruta del entorno virtual:
   ```
   C:/ruta/al/proyecto/h2o_powerbi/.venv/Scripts/python.exe
   ```

### 4.2 Probar Integración
1. Ir a "Obtener datos" → "Script de Python"
2. Pegar script de prueba:
```python
from powerbi_script import main
resultado = main(dataset)
```

## 5. Validación Final

### 5.1 Verificar Servidor H2O
```python
from src.init_h2o_server import verificar_h2o
verificar_h2o()  # Debe retornar True
```

### 5.2 Probar Modelo
```python
from src.IntegradorH2O_PBI import H2OModeloAvanzado
modelo = H2OModeloAvanzado()
resultado = modelo.entrenar(datos=datos_prueba)
```

### 5.3 Verificar Logs
```bash
# Revisar logs de operación
tail -f logs/h2o/h2o_server_*.log
tail -f logs/modelos/modelo_manager_*.log
```

## 6. Mantenimiento

### 6.1 Limpieza Regular
```bash
# Limpiar archivos temporales
python config/limpiar_todo.py

# Rotar logs antiguos (más de 30 días)
find logs/ -name "*.log" -mtime +30 -delete
```

### 6.2 Backup
```bash
# Backup de modelos y configuración
tar -czf backup_$(date +%Y%m%d).tar.gz modelos/ config/
```

### 6.3 Actualización
```bash
# Actualizar dependencias
pip install -r requirements.txt --upgrade

# Ejecutar tests post-actualización
python -m tests.test_suite
```

## 7. Troubleshooting

### 7.1 Logs de Error
```bash
# Ver errores recientes
grep ERROR logs/h2o/*.log
grep ERROR logs/modelos/*.log
```

### 7.2 Reinicio de Servicios
```python
from src.init_h2o_server import detener_servidor, iniciar_servidor_h2o

# Reiniciar H2O
detener_servidor()
iniciar_servidor_h2o()
```

### 7.3 Limpiar Cache
```bash
# Limpiar cache de H2O
rm -rf temp/h2o_temp/*
``` 