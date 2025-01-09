import logging
import os
from datetime import datetime

class Logger:
    """Clase para manejar el logging de la aplicación"""
    
    def __init__(self, nombre: str):
        """
        Inicializa el logger
        
        Args:
            nombre: Nombre del logger
        """
        self.logger = logging.getLogger(nombre)
        self.logger.setLevel(logging.INFO)
        
        # Crear directorio de logs si no existe
        self.logs_dir = os.path.abspath(os.path.join(os.getcwd(), 'logs'))
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configurar handler para archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"{nombre}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Configurar handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Crear formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Agregar handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, mensaje: str, extra: dict = None):
        """
        Registra un mensaje de nivel INFO
        
        Args:
            mensaje: Mensaje a registrar
            extra: Datos adicionales (opcional)
        """
        if extra:
            self.logger.info(f"{mensaje} - {extra}")
        else:
            self.logger.info(mensaje)
    
    def error(self, mensaje: str, exc_info=None):
        """
        Registra un mensaje de nivel ERROR
        
        Args:
            mensaje: Mensaje a registrar
            exc_info: Información de excepción (opcional)
        """
        self.logger.error(mensaje, exc_info=exc_info)
    
    def warning(self, mensaje: str):
        """
        Registra un mensaje de nivel WARNING
        
        Args:
            mensaje: Mensaje a registrar
        """
        self.logger.warning(mensaje)
    
    def debug(self, mensaje: str):
        """
        Registra un mensaje de nivel DEBUG
        
        Args:
            mensaje: Mensaje a registrar
        """
        self.logger.debug(mensaje) 