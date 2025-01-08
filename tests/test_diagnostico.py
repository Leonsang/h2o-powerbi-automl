import unittest
import pandas as pd
import h2o
import os
from src.IntegradorH2O_PBI import H2OModeloAvanzado
from src.analizar_resultados import analizar_resultados
from src.asistente_ia import AsistenteDataScience
from src.logger import Logger

logger = Logger('test_diagnostico')

class TestDiagnostico(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Preparar entorno de pruebas"""
        # Datos sintéticos
        cls.datos_prueba = pd.DataFrame({
            'x1': range(100),
            'x2': [i * 2 for i in range(100)],
            'x3': [i * 3 for i in range(100)],
            'categorica': ['A', 'B', 'C', 'D'] * 25,
            'target': [i + i * 2 + i * 3 for i in range(100)]
        })
        
        # Inicializar componentes
        h2o.init()
        cls.modelo = H2OModeloAvanzado()
        cls.asistente = AsistenteDataScience()
        
        # Entrenar modelo
        cls.resultado = cls.modelo.entrenar(
            datos=cls.datos_prueba,
            objetivo='target'
        )
        
        # Directorio para outputs
        cls.output_dir = "test_output/diagnostico"
        os.makedirs(cls.output_dir, exist_ok=True)

    def test_analisis_completo(self):
        """Verifica el análisis completo con IA"""
        try:
            # Asegurar compatibilidad de versiones
            import numpy as np
            np.random.seed(42)  # Consistente entre versiones
            
            analisis = analizar_resultados(
                datos=self.datos_prueba,
                predicciones=self.resultado['predicciones'],
                objetivo='target',
                tipo_modelo='regresion',
                output_dir=self.output_dir
            )
            
            # Verificar componentes básicos
            self.assertIn('metricas', analisis)
            self.assertIn('importancia_variables', analisis)
            self.assertIn('shap_values', analisis)
            
            # Verificar análisis IA
            self.assertIn('analisis_ia', analisis)
            self.assertIn('interpretacion_general', analisis['analisis_ia'])
            self.assertIn('recomendaciones_tecnicas', analisis['analisis_ia'])
            
            logger.info("Análisis completo generado correctamente")
            
        except Exception as e:
            logger.error(f"Error en análisis completo: {str(e)}")
            raise

    def test_explicaciones_ia(self):
        """Verifica las explicaciones de IA"""
        try:
            # Generar explicaciones
            explicaciones = self.asistente.interpretar_resultados(
                self.resultado
            )
            
            # Verificar estructura
            self.assertIsInstance(explicaciones, str)
            self.assertGreater(len(explicaciones), 100)
            
            # Verificar contenido
            terminos_esperados = [
                'análisis',
                'modelo',
                'variables',
                'recomendaciones'
            ]
            
            for termino in terminos_esperados:
                self.assertIn(termino.lower(), explicaciones.lower())
            
            logger.info("Explicaciones IA verificadas correctamente")
            
        except Exception as e:
            logger.error(f"Error en explicaciones IA: {str(e)}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Limpiar recursos"""
        h2o.cluster().show_status()
        h2o.remove_all()
        logger.info("Recursos liberados")

if __name__ == '__main__':
    unittest.main() 