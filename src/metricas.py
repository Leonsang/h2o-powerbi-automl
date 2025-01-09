import numpy as np
import pandas as pd
from sklearn.metrics import *
from src.logger import Logger

logger = Logger('metricas')

class MetricasManager:
    """Gestiona todas las métricas del modelo"""
    
    def __init__(self):
        self.logger = logger

    def calcular_metricas_clasificacion(self, y_true, y_pred, y_proba=None):
        """
        Calcula métricas para modelos de clasificación
        """
        try:
            metricas = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
            if y_proba is not None:
                metricas['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                metricas['log_loss'] = log_loss(y_true, y_proba)
            
            return metricas
        except Exception as e:
            self.logger.error(f"Error calculando métricas de clasificación: {str(e)}")
            raise

    def calcular_metricas_regresion(self, y_true, y_pred):
        """
        Calcula métricas para modelos de regresión
        """
        try:
            metricas = {
                'r2': r2_score(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mape': mean_absolute_percentage_error(y_true, y_pred),
                'explained_variance': explained_variance_score(y_true, y_pred)
            }
            return metricas
        except Exception as e:
            self.logger.error(f"Error calculando métricas de regresión: {str(e)}")
            raise

    def evaluar_cross_validation(self, modelo, X, y, cv=5, tipo_modelo='clasificacion'):
        """
        Evalúa el modelo usando validación cruzada
        """
        try:
            if tipo_modelo == 'clasificacion':
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

            scores = cross_validate(
                modelo, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )

            resultados = {
                'metricas_cv': {
                    'test': {k: np.mean(v) for k, v in scores.items() if k.startswith('test_')},
                    'train': {k: np.mean(v) for k, v in scores.items() if k.startswith('train_')}
                },
                'std_cv': {
                    'test': {k: np.std(v) for k, v in scores.items() if k.startswith('test_')},
                    'train': {k: np.std(v) for k, v in scores.items() if k.startswith('train_')}
                }
            }
            
            return resultados
        except Exception as e:
            self.logger.error(f"Error en validación cruzada: {str(e)}")
            raise

    def analizar_residuos(self, y_true, y_pred):
        """
        Analiza los residuos del modelo
        """
        try:
            residuos = y_true - y_pred
            analisis = {
                'residuos': {
                    'mean': float(np.mean(residuos)),
                    'std': float(np.std(residuos)),
                    'min': float(np.min(residuos)),
                    'max': float(np.max(residuos)),
                    'skew': float(pd.Series(residuos).skew()),
                    'kurtosis': float(pd.Series(residuos).kurtosis())
                },
                'normalidad': {
                    'shapiro': shapiro(residuos)[1],
                    'normaltest': normaltest(residuos)[1]
                }
            }
            return analisis
        except Exception as e:
            self.logger.error(f"Error analizando residuos: {str(e)}")
            raise

    def calcular_metricas_por_segmento(self, y_true, y_pred, segmentos):
        """
        Calcula métricas separadas por segmentos
        """
        try:
            metricas_segmentos = {}
            for segmento in np.unique(segmentos):
                mask = segmentos == segmento
                metricas_segmentos[segmento] = {
                    'size': np.sum(mask),
                    'metricas': self.calcular_metricas_regresion(y_true[mask], y_pred[mask])
                    if len(np.unique(y_true)) > 10
                    else self.calcular_metricas_clasificacion(y_true[mask], y_pred[mask])
                }
            return metricas_segmentos
        except Exception as e:
            self.logger.error(f"Error calculando métricas por segmento: {str(e)}")
            raise 