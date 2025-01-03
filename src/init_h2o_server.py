import h2o
import os
import time
import atexit
import tempfile
from .logger import Logger

logger = Logger('h2o_server')

# Configuración global del servidor H2O
H2O_CONFIG = {
    'url': 'http://localhost',
    'port': 54321,  # Puerto fijo
    'name': 'h2o_pbi',
    'max_mem_size': '4g',
    'ice_root': os.path.join(tempfile.gettempdir(), 'h2o_temp'),
    'log_level': 'ERRR',
    'nthreads': -1
}

def iniciar_servidor_h2o():
    """Inicia el servidor H2O con la configuración establecida"""
    try:
        logger.info("Iniciando servidor H2O")
        h2o.init(
            port=54322,  # Puerto por defecto
            nthreads=-1, # Usar todos los núcleos
            ice_root=os.path.join(os.getcwd(), "h2o_temp"),
            name="h2o_server"
        )
        return True
    except Exception as e:
        logger.error(f"Error iniciando H2O: {str(e)}")
        return False

def verificar_h2o():
    """Verifica si H2O está activo y respondiendo"""
    try:
        if h2o.connection():
            h2o.cluster().show_status()
            return True
        return False
    except:
        return False

def detener_servidor():
    """Detiene el servidor H2O de forma segura"""
    try:
        logger.info("Deteniendo servidor H2O")
        h2o.cluster().shutdown()
    except Exception as e:
        logger.error(f"Error deteniendo H2O: {str(e)}")

# Registrar función para detener el servidor al cerrar
atexit.register(detener_servidor)

# Iniciar el servidor si este script se ejecuta directamente
if __name__ == "__main__":
    iniciar_servidor_h2o() 