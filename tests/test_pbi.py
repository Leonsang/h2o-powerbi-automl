import unittest
import pandas as pd
import h2o
from src.IntegradorH2O_PBI import H2OModeloAvanzado
from src.logger import Logger

logger = Logger('test_pbi')

class TestPowerBI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Preparar datos de prueba"""
        cls.datos_prueba = pd.DataFrame({
            'x1': range(100),
            'x2': [i * 2 for i in range(100)],
            'target': [i + i * 2 for i in range(100)]
        })
        h2o.init()

    def test_integracion_powerbi(self):
        """Verifica la integración básica con Power BI"""
        try:
            # 1. Crear modelo
            modelo = H2OModeloAvanzado(
                objetivo='target',
                predictoras=['x1', 'x2']
            )
            
            # 2. Entrenar con datos de prueba
            resultado = modelo.entrenar(self.datos_prueba)
            self.assertTrue(resultado['modelo'])
            self.assertTrue(resultado['predicciones'] is not None)
            
            logger.info("Integración con Power BI exitosa")
            
        except Exception as e:
            logger.error(f"Error en integración Power BI: {str(e)}")
            raise

    def test_predicciones_powerbi(self):
        """Verifica las predicciones para Power BI"""
        try:
            modelo = H2OModeloAvanzado(
                objetivo='target',
                predictoras=['x1', 'x2']
            )
            
            # 1. Entrenar modelo
            resultado = modelo.entrenar(self.datos_prueba)
            
            # 2. Hacer predicciones
            nuevos_datos = pd.DataFrame({
                'x1': range(10),
                'x2': range(10)
            })
            predicciones = modelo.predecir(
                modelo=resultado['modelo'],
                datos=nuevos_datos
            )
            
            # 3. Verificar predicciones
            self.assertEqual(len(predicciones), len(nuevos_datos))
            self.assertTrue(all(isinstance(x, (int, float)) for x in predicciones))
            
            logger.info("Predicciones para Power BI generadas correctamente")
            
        except Exception as e:
            logger.error(f"Error en predicciones Power BI: {str(e)}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Limpiar recursos"""
        h2o.cluster().show_status()
        h2o.remove_all()

if __name__ == '__main__':
    unittest.main() 