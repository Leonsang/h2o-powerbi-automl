import unittest
import pandas as pd
import h2o
from src.IntegradorH2O_PBI import H2OModeloAvanzado
from src.asistente_ia import AsistenteDataScience
from src.logger import Logger

logger = Logger('test_integracion')

class TestIntegracion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Preparar entorno de pruebas"""
        cls.datos_prueba = pd.DataFrame({
            'numerica1': range(100),
            'numerica2': [i * 2 for i in range(100)],
            'categorica': ['A', 'B'] * 50,
            'objetivo': [i + i * 2 for i in range(100)]
        })
        h2o.init()
        cls.asistente = AsistenteDataScience()
        logger.info("Entorno de pruebas inicializado")

    def test_flujo_completo(self):
        """Prueba el flujo completo con IA"""
        try:
            # 1. Inicialización
            modelo = H2OModeloAvanzado()
            
            # 2. Análisis y entrenamiento
            resultado = modelo.entrenar(
                datos=self.datos_prueba,
                objetivo='objetivo'
            )
            
            # 3. Interpretabilidad IA
            analisis_ia = self.asistente.interpretar_resultados(resultado)
            recomendaciones = self.asistente.generar_recomendaciones_tecnicas(resultado)
            
            # 4. Validación de componentes
            self.assertIn('modelo', resultado)
            self.assertIn('predicciones', resultado)
            self.assertIsInstance(analisis_ia, str)
            self.assertIsInstance(recomendaciones, str)
            
            logger.info("Flujo completo ejecutado correctamente")
            
        except Exception as e:
            logger.error(f"Error en flujo completo: {str(e)}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Limpiar recursos"""
        h2o.cluster().show_status()
        h2o.remove_all()
        logger.info("Recursos liberados")

if __name__ == '__main__':
    unittest.main() 