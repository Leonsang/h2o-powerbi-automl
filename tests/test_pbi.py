import unittest
import pandas as pd
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.IntegradorH2O_PBI import H2OModeloAvanzado, main
from src.init_h2o_server import iniciar_servidor_h2o, detener_servidor, H2O_CONFIG

class TestPowerBI(unittest.TestCase):
    def setUp(self):
        """Preparar el entorno para cada test"""
        self.servidor_iniciado = iniciar_servidor_h2o()
        self.datos = pd.read_csv('datos/matriculas.csv')

    def tearDown(self):
        """Limpiar después de cada test"""
        detener_servidor()

    def test_script_pbi(self):
        """Simula la ejecución desde Power BI"""
        try:
            # Probar función principal
            resultado = main(dataset=self.datos)
            
            # Verificaciones
            self.assertIsInstance(resultado, pd.DataFrame)
            self.assertIn('prediccion', resultado.columns)
            self.assertNotIn('Error', resultado.columns)
            
            # Verificar que usamos el puerto correcto
            self.assertTrue(h2o.connection().current_connection_url.endswith(str(H2O_CONFIG['port'])))
            
        except Exception as e:
            self.fail(f"El script falló: {str(e)}")

if __name__ == '__main__':
    unittest.main() 