import os
from pathlib import Path

def crear_estructura_proyecto():
    """Crea la estructura base del proyecto"""
    
    # Estructura de directorios
    directorios = [
        'modelos',
        'datos',
        'logs/h2o',
        'logs/modelos',
        'logs/tests',
        'temp/h2o_temp',
        'src',
        'config',
        'tests'
    ]
    
    # Crear directorios
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)
        gitkeep = Path(dir) / '.gitkeep'
        gitkeep.touch(exist_ok=True)
    
    # Crear README.md en datos si no existe
    datos_readme = Path('datos/README.md')
    if not datos_readme.exists():
        with open(datos_readme, 'w', encoding='utf-8') as f:
            f.write("""# Datos de ejemplo
Este directorio contiene los datos de ejemplo y pruebas para el proyecto.

## Archivos
- matriculas.csv: Datos de ejemplo para predicción de matrículas
- ventas_simuladas.csv: Datos simulados para pruebas""")

    print("✅ Estructura de directorios creada:")
    for dir in directorios:
        print(f"  └── {dir}/")

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