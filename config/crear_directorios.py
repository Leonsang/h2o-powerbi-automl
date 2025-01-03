import os
from pathlib import Path

def crear_estructura_proyecto():
    """Crea la estructura base del proyecto"""
    
    # Estructura de directorios
    directorios = [
        'modelos',          # Directorio principal para modelos
        'logs',            # Logs del sistema
        'logs/h2o',        # Logs específicos de H2O
        'logs/modelos',    # Logs de operaciones con modelos
        'logs/tests',      # Logs de tests
        'config',          # Configuraciones
        'temp/h2o_temp'    # Archivos temporales de H2O
    ]
    
    # Crear directorios
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)
        
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