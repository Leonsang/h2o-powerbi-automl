"""
Creación y gestión de la estructura de directorios del proyecto H2O AutoML para Power BI.

Alcances:
- Crea la estructura base necesaria para el proyecto
- Mantiene directorios vacíos con .gitkeep
- Genera documentación inicial en datos/

Limitaciones:
- Solo crea estructura local
- No gestiona permisos de directorios
- No sincroniza con repositorios remotos
"""

import os
from pathlib import Path
from typing import List, Dict

# Constantes de configuración
DIRECTORIOS_BASE: List[str] = [
    'modelos',      # Almacenamiento de modelos entrenados
    'datos',        # Datos de ejemplo y pruebas
    'logs/h2o',     # Logs específicos del servidor H2O
    'logs/modelos', # Logs de entrenamiento y predicción
    'logs/tests',   # Logs de pruebas unitarias
    'temp/h2o_temp',# Archivos temporales de H2O
    'src',          # Código fuente del proyecto
    'config',       # Archivos de configuración
    'tests'         # Tests unitarios y de integración
]

# Descripción de uso de directorios
USO_DIRECTORIOS: Dict[str, str] = {
    'modelos': 'Modelos H2O serializados y metadata',
    'datos': 'Datasets de ejemplo y pruebas',
    'logs': 'Logs del sistema por componente',
    'temp': 'Archivos temporales (limpieza automática)',
    'src': 'Código fuente principal',
    'config': 'Configuraciones y scripts de instalación',
    'tests': 'Suite completa de pruebas'
}

def crear_estructura_proyecto() -> None:
    """
    Crea la estructura base del proyecto.
    
    Estructura:
    proyecto/
    ├── modelos/          # Modelos entrenados
    ├── datos/            # Datos de ejemplo
    ├── logs/            
    │   ├── h2o/         # Logs de H2O
    │   ├── modelos/     # Logs de modelos
    │   └── tests/       # Logs de tests
    ├── temp/
    │   └── h2o_temp/    # Temp de H2O
    ├── src/             # Código fuente
    ├── config/          # Configuración
    └── tests/           # Tests
    
    Returns:
        None
    """
    try:
        # Crear directorios base
        for dir in DIRECTORIOS_BASE:
            Path(dir).mkdir(parents=True, exist_ok=True)
            gitkeep = Path(dir) / '.gitkeep'
            gitkeep.touch(exist_ok=True)
        
        # Crear README.md en datos
        datos_readme = Path('datos/README.md')
        if not datos_readme.exists():
            with open(datos_readme, 'w', encoding='utf-8') as f:
                f.write("""# 📊 Datos de Ejemplo

Este directorio contiene los datos de ejemplo y pruebas para el proyecto H2O AutoML.

## 📋 Archivos
- `matriculas.csv`: Datos históricos de matrículas para predicción
- `ventas_simuladas.csv`: Dataset simulado para pruebas de integración

## 🔒 Uso Local
Estos datos son solo para pruebas locales. No subir datos sensibles o productivos.

## 📝 Notas
- Máximo 100MB por archivo
- Formatos soportados: CSV, XLSX
- Encoding: UTF-8
- Separador: coma (,)
""")
        
        print("\n✅ Estructura de directorios creada:")
        for dir in DIRECTORIOS_BASE:
            print(f"  └── {dir:<15} # {USO_DIRECTORIOS.get(dir.split('/')[0], '')}")
            
    except Exception as e:
        print(f"\n❌ Error creando directorios: {str(e)}")
        raise

def limpiar_directorios_temp():
    """Limpia directorios temporales"""
    try:
        temp_dir = Path('temp/h2o_temp')
        if temp_dir.exists():
            for file in temp_dir.glob('*'):
                file.unlink()
            print("✅ Directorio temporal limpiado")
    except Exception as e:
        print(f"❌ Error limpiando temporales: {str(e)}")

if __name__ == "__main__":
    crear_estructura_proyecto() 