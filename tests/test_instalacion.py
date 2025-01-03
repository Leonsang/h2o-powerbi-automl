import unittest
import h2o
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.init_h2o_server import iniciar_servidor_h2o, detener_servidor, H2O_CONFIG
from src.logger import Logger

logger = Logger('test_instalacion')

class TestInstalacion(unittest.TestCase):
    def setUp(self):
        """Preparar el entorno para cada test"""
        logger.info("Iniciando test de instalación")
        self.servidor_iniciado = iniciar_servidor_h2o()

    def tearDown(self):
        """Limpiar después de cada test"""
        logger.info("Finalizando test de instalación")
        detener_servidor()

    def test_servidor_h2o(self):
        """Verifica que el servidor H2O se inicia correctamente"""
        logger.info("Verificando servidor H2O")
        self.assertTrue(self.servidor_iniciado)
        self.assertTrue(h2o.connection())
        
        # Verificar configuración
        status = h2o.cluster().show_status()
        self.assertEqual(status['H2O_cluster_name'], H2O_CONFIG['name'])
        self.assertTrue(h2o.connection().current_connection_url.endswith(str(H2O_CONFIG['port'])))

    def test_conexion_h2o(self):
        """Verifica que podemos conectar al servidor H2O"""
        try:
            h2o.connect(url=f"{H2O_CONFIG['url']}:{H2O_CONFIG['port']}")
            self.assertTrue(True)
        except:
            self.fail("No se pudo conectar al servidor H2O")

if __name__ == '__main__':
    unittest.main() 