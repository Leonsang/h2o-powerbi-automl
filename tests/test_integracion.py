import unittest
import pandas as pd
import h2o
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.IntegradorH2O_PBI import H2OModeloAvanzado
from src.init_h2o_server import iniciar_servidor_h2o, detener_servidor, H2O_CONFIG
from src.logger import Logger

logger = Logger('test_integracion')

class TestIntegracion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Preparar el entorno una vez para todos los tests"""
        logger.info("Iniciando suite de tests de integración")
        cls.servidor_iniciado = iniciar_servidor_h2o()
        if not cls.servidor_iniciado:
            logger.error("No se pudo iniciar el servidor H2O")
            raise Exception("No se pudo iniciar el servidor H2O")
            
        # Cargar datos de prueba
        try:
            cls.datos = pd.read_csv('datos/matriculas.csv')
        except Exception as e:
            raise Exception(f"Error cargando datos: {str(e)}")
            
        cls.modelo = H2OModeloAvanzado()

    @classmethod
    def tearDownClass(cls):
        """Limpiar después de todos los tests"""
        detener_servidor()

    def test_conexion_activa(self):
        """Verifica que la conexión H2O está activa"""
        self.assertTrue(h2o.connection())
        self.assertEqual(h2o.cluster().show_status()['H2O_cluster_name'], H2O_CONFIG['name'])

    def test_entrenamiento_modelo(self):
        """Verifica el entrenamiento del modelo"""
        resultado = self.modelo.entrenar(
            datos=self.datos,
            columna_objetivo='matriculas'
        )
        self.assertIsInstance(resultado, dict)
        self.assertIn('predicciones', resultado)
        self.assertIsInstance(resultado['predicciones'], pd.DataFrame)

if __name__ == '__main__':
    unittest.main() 