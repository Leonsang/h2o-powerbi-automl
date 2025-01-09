import h2o
import os
import time
import atexit
import tempfile
from pathlib import Path
import logging

# Configurar logger básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('h2o_server')

# Configuración H2O
H2O_CONFIG = {
    'name': 'h2o_pbi',
    'url': 'http://localhost',
    'port': 54321
}

def iniciar_servidor_h2o():
    """Inicia el servidor H2O"""
    try:
        h2o.init(
            name=H2O_CONFIG['name'],
            port=H2O_CONFIG['port']
        )
        return True
    except Exception as e:
        logger.error(f"Error iniciando H2O: {str(e)}")
        return False

def detener_servidor():
    """Detiene el servidor H2O"""
    try:
        h2o.cluster().shutdown()
        return True
    except:
        return False

if __name__ == '__main__':
    iniciar_servidor_h2o() 