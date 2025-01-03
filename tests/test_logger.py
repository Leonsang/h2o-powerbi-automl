import unittest
import os
import shutil
from datetime import datetime
from src.logger import Logger

class TestLogger(unittest.TestCase):
    def setUp(self):
        """Preparar el entorno para cada test"""
        self.log_dir = 'logs'
        self.logger = Logger('test_logger')
        self.log_file = f"logs/test_logger_{datetime.now().strftime('%Y%m%d')}.log"

    def tearDown(self):
        """Limpiar después de cada test"""
        # Eliminar archivos de log de prueba
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_crear_directorio(self):
        """Verifica que se crea el directorio de logs"""
        self.assertTrue(os.path.exists(self.log_dir))

    def test_crear_archivo_log(self):
        """Verifica que se crea el archivo de log"""
        self.logger.info("Test message")
        self.assertTrue(os.path.exists(self.log_file))

    def test_niveles_log(self):
        """Verifica que se registran diferentes niveles de log"""
        test_messages = {
            'info': "Test info message",
            'warning': "Test warning message",
            'error': "Test error message",
            'debug': "Test debug message"
        }
        
        # Escribir mensajes
        self.logger.info(test_messages['info'])
        self.logger.warning(test_messages['warning'])
        self.logger.error(test_messages['error'])
        self.logger.debug(test_messages['debug'])
        
        # Verificar contenido
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn(test_messages['info'], content)
            self.assertIn(test_messages['warning'], content)
            self.assertIn(test_messages['error'], content)
            # Debug no debería aparecer por default
            self.assertNotIn(test_messages['debug'], content)

    def test_formato_log(self):
        """Verifica el formato correcto del log"""
        test_message = "Test format message"
        self.logger.info(test_message)
        
        with open(self.log_file, 'r') as f:
            line = f.readline()
            # Verificar componentes del formato
            self.assertRegex(line, r'\d{4}-\d{2}-\d{2}')  # Fecha
            self.assertIn('test_logger', line)             # Nombre logger
            self.assertIn('INFO', line)                    # Nivel
            self.assertIn(test_message, line)              # Mensaje

    def test_multiples_loggers(self):
        """Verifica que múltiples loggers funcionan independientemente"""
        logger1 = Logger('test_logger1')
        logger2 = Logger('test_logger2')
        
        msg1 = "Message from logger1"
        msg2 = "Message from logger2"
        
        logger1.info(msg1)
        logger2.info(msg2)
        
        # Verificar archivos separados
        self.assertTrue(os.path.exists(f"logs/test_logger1_{datetime.now().strftime('%Y%m%d')}.log"))
        self.assertTrue(os.path.exists(f"logs/test_logger2_{datetime.now().strftime('%Y%m%d')}.log")) 