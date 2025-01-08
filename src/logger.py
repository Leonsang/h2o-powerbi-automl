import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, nombre):
        # Crear directorio de logs
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configurar logger
        self.logger = logging.getLogger(nombre)
        self.logger.setLevel(logging.INFO)
        
        # Handler para archivo
        fecha = datetime.now().strftime("%Y%m%d_%H%M")
        fh = logging.FileHandler(
            os.path.join(self.log_dir, f"{nombre}_{fecha}.log")
        )
        fh.setLevel(logging.INFO)
        
        # Handler para consola
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Agregar handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, msg):
        self.logger.info(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def debug(self, msg):
        self.logger.debug(msg) 