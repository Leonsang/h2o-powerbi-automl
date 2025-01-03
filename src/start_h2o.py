import h2o
import os
import time
from .init_h2o_server import iniciar_servidor_h2o

def iniciar_h2o(config=None):
    """Conecta con el servidor H2O existente o inicia uno nuevo"""
    try:
        # Intentar conectar con servidor existente
        if h2o.connection():
            try:
                h2o.cluster().show_status()
                print("Conectado a servidor H2O existente")
                return True
            except:
                print("Servidor H2O no responde, iniciando nuevo servidor...")
        
        # Iniciar nuevo servidor si no hay uno activo
        return iniciar_servidor_h2o()

    except Exception as e:
        print(f"Error conectando con H2O: {str(e)}")
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

def cerrar_h2o():
    """Desconecta del servidor H2O sin detenerlo"""
    try:
        if h2o.connection():
            h2o.connection().close()
            return True
    except Exception as e:
        print(f"Error al desconectar de H2O: {str(e)}")
    return False 