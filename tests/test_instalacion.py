import unittest
import h2o
from src.init_h2o_server import H2O_CONFIG
from src.logger import Logger

logger = Logger('test_instalacion')

class TestInstalacion(unittest.TestCase):
    def test_conexion_h2o(self):
        """Verifica que podemos conectar al servidor H2O"""
        # Conectar al servidor
        conn = h2o.connect(url=f"{H2O_CONFIG['url']}:{H2O_CONFIG['port']}")
        self.assertTrue(conn)
        
        # Obtener información del cluster
        cluster = h2o.cluster()
        self.assertTrue(cluster)
        
        # Verificar estado usando show_status()
        status = cluster.show_status()
        self.assertEqual(status["H2O_cluster_name"], H2O_CONFIG['name'])
        self.assertEqual(status["H2O_cluster_status"], "locked, healthy")

    def test_servidor_h2o(self):
        """Verifica que el servidor H2O se inicia correctamente"""
        self.assertTrue(h2o.cluster())

if __name__ == '__main__':
    unittest.main() 