import subprocess
import sys
import os
from pathlib import Path
from .logger import Logger

logger = Logger('java_check')

def verificar_java():
    """
    Verifica la instalación de Java y su versión
    Returns:
        dict: Estado de Java con detalles
    """
    estado = {
        'instalado': False,
        'version': None,
        'path': None,
        'error': None
    }
    
    try:
        # 1. Verificar comando java
        resultado = subprocess.run(
            ['java', '-version'], 
            capture_output=True, 
            text=True, 
            stderr=subprocess.PIPE
        )
        
        if resultado.returncode == 0:
            estado['instalado'] = True
            # Extraer versión del output
            version_line = resultado.stderr.split('\n')[0]
            estado['version'] = version_line
            
            # Obtener path de Java
            if sys.platform == 'win32':
                java_path = subprocess.run(
                    ['where', 'java'], 
                    capture_output=True, 
                    text=True
                ).stdout.strip()
            else:
                java_path = subprocess.run(
                    ['which', 'java'], 
                    capture_output=True, 
                    text=True
                ).stdout.strip()
                
            estado['path'] = java_path
            logger.info(f"Java encontrado: {version_line}")
            
        else:
            estado['error'] = "Java no encontrado en el sistema"
            logger.error(estado['error'])
            
    except Exception as e:
        estado['error'] = f"Error verificando Java: {str(e)}"
        logger.error(estado['error'])
        
    return estado

def verificar_requisitos():
    """
    Verifica todos los requisitos para H2O
    Returns:
        bool: True si cumple todos los requisitos
    """
    # 1. Verificar Java
    java = verificar_java()
    if not java['instalado']:
        logger.error("❌ Java no instalado")
        print("""
        Por favor instale Java 8 o superior:
        1. Descargue de: https://www.java.com/download/
        2. Ejecute el instalador
        3. Reinicie el sistema
        """)
        return False
        
    # 2. Verificar versión mínima (Java 8)
    if java['version'] and '1.8' not in java['version'] and 'java version "8' not in java['version']:
        logger.warning("⚠️ Se recomienda Java 8")
        
    # 3. Verificar permisos de directorios
    temp_dir = Path(os.environ.get('TEMP', '/tmp'))
    if not os.access(temp_dir, os.W_OK):
        logger.error(f"❌ Sin permisos de escritura en {temp_dir}")
        return False
        
    logger.info("✅ Requisitos verificados correctamente")
    return True

if __name__ == "__main__":
    if verificar_requisitos():
        print("✅ Sistema listo para H2O")
    else:
        print("❌ Por favor revise los requisitos") 