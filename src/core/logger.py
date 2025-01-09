import logging
import os
import json
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

class LogMonitor:
    """Monitor de logs para alertas y estadísticas"""
    
    def __init__(self, config):
        self.config = config
        self.alert_patterns = config.get('alerts', {}).get('critical_patterns', [])
        self.enabled = config.get('alerts', {}).get('enabled', False)
        self.notification_methods = config.get('alerts', {}).get('notification_methods', [])
        self.stats = {
            'error_count': 0,
            'warning_count': 0,
            'critical_count': 0,
            'last_error': None,
            'last_critical': None
        }
    
    def check_message(self, level, message):
        """Verifica si un mensaje necesita generar alertas"""
        if not self.enabled:
            return
            
        if level in ['ERROR', 'CRITICAL']:
            self.stats[f'{level.lower()}_count'] += 1
            self.stats[f'last_{level.lower()}'] = {
                'timestamp': datetime.now().isoformat(),
                'message': message
            }
            
            for pattern in self.alert_patterns:
                if pattern in message:
                    self._send_alert(level, message)
    
    def _send_alert(self, level, message):
        """Envía alertas por los métodos configurados"""
        alert = f"ALERTA {level}: {message}"
        
        for method in self.notification_methods:
            if method == 'console':
                print(f"\033[91m{alert}\033[0m")  # Rojo para alertas
            elif method == 'file':
                with open('logs/alerts.log', 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - {alert}\n")

class Logger:
    """Sistema de logging con monitoreo y alertas"""
    
    def __init__(self, nombre, config_file=None):
        # Cargar configuración
        self.config = self._load_config(config_file)
        self.monitor = LogMonitor(self.config)
        
        # Crear directorio de logs
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Configurar logger
        self.logger = logging.getLogger(nombre)
        self.logger.setLevel(self._get_logger_level(nombre))
        
        # Limpiar handlers existentes
        self.logger.handlers = []
        
        # Configurar handlers según configuración
        self._setup_handlers(nombre)
    
    def _load_config(self, config_file):
        """Carga configuración desde archivo o usa valores por defecto"""
        if config_file is None:
            config_file = Path(__file__).parent / 'config' / 'logging_config.json'
            
        config = {
            'log_dir': 'logs',
            'max_bytes': 10485760,
            'backup_count': 5,
            'default_level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S'
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                config.update(custom_config)
            except Exception as e:
                print(f"Error cargando configuración: {str(e)}")
                
        return config
    
    def _get_logger_level(self, nombre):
        """Obtiene el nivel de log específico para el logger"""
        logger_config = self.config.get('loggers', {}).get(nombre, {})
        return getattr(logging, logger_config.get('level', self.config['default_level']))
    
    def _setup_handlers(self, nombre):
        """Configura los handlers según la configuración"""
        handlers_config = self.config.get('handlers', {})
        
        # Handler para archivo con rotación
        if handlers_config.get('file', {}).get('enabled', True):
            fecha = datetime.now().strftime("%Y%m%d_%H%M")
            fh = RotatingFileHandler(
                os.path.join(self.config['log_dir'], f"{nombre}_{fecha}.log"),
                maxBytes=self.config['max_bytes'],
                backupCount=self.config['backup_count']
            )
            fh.setLevel(getattr(logging, handlers_config['file'].get('level', 'DEBUG')))
            fh.setFormatter(self._get_formatter())
            self.logger.addHandler(fh)
        
        # Handler para consola
        if handlers_config.get('console', {}).get('enabled', True):
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, handlers_config['console'].get('level', 'INFO')))
            ch.setFormatter(self._get_formatter())
            self.logger.addHandler(ch)
    
    def _get_formatter(self):
        """Obtiene el formateador según la configuración"""
        return logging.Formatter(
            self.config['format'],
            datefmt=self.config['date_format']
        )
    
    def set_level(self, level):
        """Cambia el nivel de logging dinámicamente"""
        nivel = getattr(logging, level.upper())
        self.logger.setLevel(nivel)
        for handler in self.logger.handlers:
            handler.setLevel(nivel)
    
    def info(self, msg, extra=None):
        """Log a nivel INFO con contexto opcional"""
        self._log('info', msg, extra)
    
    def error(self, msg, extra=None, exc_info=None):
        """Log a nivel ERROR con contexto opcional y excepción"""
        if exc_info:
            msg = f"{msg}\n{self._format_exception(exc_info)}"
        self._log('error', msg, extra)
        self.monitor.check_message('ERROR', msg)
    
    def warning(self, msg, extra=None):
        """Log a nivel WARNING con contexto opcional"""
        self._log('warning', msg, extra)
    
    def debug(self, msg, extra=None):
        """Log a nivel DEBUG con contexto opcional"""
        self._log('debug', msg, extra)
    
    def critical(self, msg, extra=None, exc_info=None):
        """Log a nivel CRITICAL con contexto opcional y excepción"""
        if exc_info:
            msg = f"{msg}\n{self._format_exception(exc_info)}"
        self._log('critical', msg, extra)
        self.monitor.check_message('CRITICAL', msg)
    
    def exception(self, msg, extra=None):
        """Log una excepción con su traceback completo"""
        self.error(msg, extra, exc_info=True)
    
    def _log(self, level, msg, extra=None):
        """Método interno para logging con contexto"""
        if extra:
            msg = f"{msg} - Contexto: {json.dumps(extra)}"
        getattr(self.logger, level)(msg)
    
    def _format_exception(self, exc_info):
        """Formatea una excepción para logging"""
        if isinstance(exc_info, BaseException):
            return ''.join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
        elif isinstance(exc_info, tuple):
            return ''.join(traceback.format_exception(*exc_info))
        return str(exc_info)
    
    def get_stats(self):
        """Obtiene estadísticas del monitor de logs"""
        return self.monitor.stats 