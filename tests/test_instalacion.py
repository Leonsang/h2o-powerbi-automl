import unittest
import h2o
import pandas as pd
import numpy as np
from src.init_h2o_server import H2O_CONFIG
from src.logger import Logger
from src.modelo_manager_ia import ModeloManagerIA
from src.asistente_ia import AsistenteDataScience

logger = Logger('test_instalacion')

class TestInstalacion(unittest.TestCase):
    def test_conexion_h2o(self):
        """Verifica la conexión al servidor H2O"""
        try:
            # 1. Crear datos de prueba simple
            df = pd.DataFrame({
                'x': range(10),
                'y': range(10)
            })
            
            # 2. Conectar y subir datos
            h2o.connect(url=f"{H2O_CONFIG['url']}:{H2O_CONFIG['port']}")
            h2o_df = h2o.H2OFrame(df)
            
            # 3. Verificar operaciones básicas
            self.assertEqual(h2o_df.shape, (10, 2))
            self.assertEqual(h2o_df.names, ['x', 'y'])
            
            logger.info("Conexión a H2O exitosa - Prueba de datos completada")
            
        except Exception as e:
            logger.error(f"Error conectando a H2O: {str(e)}")
            raise

    def test_modelo_ia(self):
        """Verifica la instalación del modelo de IA"""
        try:
            # 1. Verificar gestor de modelos
            modelo_manager = ModeloManagerIA()
            self.assertTrue(modelo_manager.verificar_modelo())
            
            # 2. Verificar asistente IA
            asistente = AsistenteDataScience()
            self.assertIsNotNone(asistente.llm)
            
            logger.info("Componentes IA verificados correctamente")
            
        except Exception as e:
            logger.error(f"Error verificando IA: {str(e)}")
            raise

    def test_dependencias(self):
        """Verifica todas las dependencias"""
        try:
            # Core
            import h2o
            import pandas as pd
            import numpy as np
            
            # ML y análisis
            import sklearn
            import shap
            import lime
            import dice_ml
            
            # IA y LLM
            import langchain
            import gpt4all
            
            # Visualización
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            versions = {
                'h2o': '3.46.0.1',
                'pandas': '2.0.3',
                'numpy': '1.24.3',
                'sklearn': '1.3.0',
                'shap': '0.41.0',
                'langchain': '0.0.200'
            }
            
            for lib, expected_version in versions.items():
                module = globals()[lib]
                actual_version = module.__version__
                self.assertEqual(actual_version, expected_version, 
                    f"Versión incorrecta de {lib}: esperada {expected_version}, actual {actual_version}")
            
            logger.info("Todas las dependencias instaladas correctamente")
            
        except ImportError as e:
            logger.error(f"Falta dependencia: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main() 