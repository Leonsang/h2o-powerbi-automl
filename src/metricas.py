import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from src.logger import Logger

logger = Logger('metricas')

class Metricas:
    def calcular_metricas_globales(self, reales, predicciones, tipo_modelo='regresion'):
        """Calcula métricas según tipo de modelo"""
        try:
            if tipo_modelo == 'regresion':
                return self.metricas_regresion(reales, predicciones)
            else:
                return self.metricas_clasificacion(reales, predicciones)
        except Exception as e:
            logger.error(f"Error calculando métricas: {str(e)}")
            raise

    def metricas_regresion(self, reales, predicciones):
        """Métricas para modelos de regresión"""
        try:
            rmse = np.sqrt(mean_squared_error(reales, predicciones))
            r2 = r2_score(reales, predicciones)
            mae = np.mean(np.abs(reales - predicciones))
            
            return {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'explicacion': {
                    'rmse': 'Error cuadrático medio (menor es mejor)',
                    'r2': 'Coeficiente de determinación (más cercano a 1 es mejor)',
                    'mae': 'Error absoluto medio (menor es mejor)'
                }
            }
        except Exception as e:
            logger.error(f"Error en métricas regresión: {str(e)}")
            raise

    def metricas_clasificacion(self, reales, predicciones):
        """Métricas para modelos de clasificación"""
        try:
            accuracy = accuracy_score(reales, predicciones)
            precision = precision_score(reales, predicciones, average='weighted')
            recall = recall_score(reales, predicciones, average='weighted')
            f1 = f1_score(reales, predicciones, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'explicacion': {
                    'accuracy': 'Precisión general del modelo',
                    'precision': 'Precisión por clase',
                    'recall': 'Exhaustividad por clase',
                    'f1': 'Media armónica de precisión y recall'
                }
            }
        except Exception as e:
            logger.error(f"Error en métricas clasificación: {str(e)}")
            raise 