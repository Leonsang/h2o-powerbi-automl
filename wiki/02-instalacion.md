# 🛠️ Instalación y Configuración

## 📋 Requisitos del Sistema

### Software
- Python 3.8+
- Java 8+ (requerido por H2O)
- Power BI Desktop
- Git (opcional)

### Hardware Recomendado
- RAM: 8GB mínimo, 16GB recomendado
- CPU: 4 cores mínimo
- Espacio: 2GB libres

## 🚀 Proceso de Instalación

### Windows
```bash
# 1. Clonar repositorio
git clone https://github.com/Leonsang/h2o-powerbi-automl.git
cd h2o-powerbi-automl

# 2. Ejecutar instalador
install.bat
```

### Linux/Mac
```bash
# 1. Clonar repositorio
git clone https://github.com/Leonsang/h2o-powerbi-automl.git
cd h2o-powerbi-automl

# 2. Dar permisos y ejecutar
chmod +x install.sh
./install.sh
```

## ⚙️ Configuración Inicial

1. **Configuración de H2O**
   ```python
   # config/h2o_config.json
   {
     "max_models": 20,
     "max_runtime_secs": 300,
     "seed": 1
   }
   ```

2. **Variables de Entorno**
   ```bash
   H2O_JAVA_XMX=4g
   H2O_PORT=54321
   ```

## ✅ Verificación

1. **Test de Instalación**
   ```bash
   python -m tests.test_instalacion
   ```

2. **Verificar Conexión H2O**
   ```python
   from src.init_h2o_server import verificar_conexion
   verificar_conexion()
   ```

## 🔍 Troubleshooting

### Problemas Comunes

1. **Error: H2O no inicia**
   - Verificar Java instalado
   - Revisar puertos disponibles
   - Comprobar memoria RAM

2. **Error: Módulos no encontrados**
   - Reinstalar dependencias
   - Verificar PATH
   - Actualizar pip 