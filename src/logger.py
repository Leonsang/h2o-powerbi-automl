import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, nombre='h2o_pbi'):
        """Inicializa el sistema de logs"""
        self.logger = logging.getLogger(nombre)
        self.logger.setLevel(logging.INFO)
        
        # Determinar subdirectorio según el tipo de logger
        if 'test' in nombre:
            subdir = 'logs/tests'
        elif 'h2o' in nombre:
            subdir = 'logs/h2o'
        elif 'modelo' in nombre:
            subdir = 'logs/modelos'
        else:
            subdir = 'logs'
        
        # Crear directorio si no existe
        os.makedirs(subdir, exist_ok=True)
        
        # Handler para archivo
        log_file = f"{subdir}/{nombre}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Agregar handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, mensaje):
        self.logger.info(mensaje)
    
    def warning(self, mensaje):
        self.logger.warning(mensaje)
    
    def error(self, mensaje):
        self.logger.error(mensaje)
    
    def debug(self, mensaje):
        self.logger.debug(mensaje) 