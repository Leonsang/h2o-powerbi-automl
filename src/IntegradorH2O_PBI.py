import pandas as pd
import h2o
import os
from datetime import datetime
from .modelo_manager import ModeloManager
from .init_h2o_server import iniciar_servidor_h2o, detener_servidor
from .analizar_resultados import analizar_resultados
from .logger import Logger

class H2OModeloAvanzado:
    def __init__(self):
        """Inicializa H2O con todas sus capacidades pero de forma simple"""
        self.modelo_manager = ModeloManager()
        self.configuracion = {
            'max_runtime_secs': 300,
            'max_models': 20,
            'seed': 1,
            'nfolds': 5,
            'balance_classes': True,
            'keep_cross_validation_predictions': True,
            'sort_metric': 'auto',
            'stopping_metric': 'auto',
            'stopping_rounds': 3,
            'stopping_tolerance': 0.001,
            'include_algos': [  # Incluir todos los algoritmos disponibles
                'DRF', 'GLM', 'XGBoost', 'GBM', 'DeepLearning',
                'StackedEnsemble'
            ]
        }
        self.logger = Logger('h2o_modelo')
        self.logger.info("Iniciando H2O Modelo Avanzado")

    def obtener_metricas_modelo(self, modelo, datos):
        """Obtiene métricas completas según el tipo de modelo"""
        metricas = {}
        
        # Métricas básicas
        perf = modelo.model_performance(datos)
        metricas['basic'] = {
            'rmse': perf.rmse(),
            'mse': perf.mse(),
            'mae': perf.mae(),
            'rmsle': perf.rmsle(),
            'r2': perf.r2()
        }
        
        # Métricas específicas por tipo de modelo
        if modelo.__class__.__name__ == 'H2OGradientBoostingEstimator':
            metricas['gbm'] = {
                'variable_importance': modelo.varimp(use_pandas=True),
                'scoring_history': modelo.scoring_history(),
                'model_summary': modelo.summary()
            }
        elif modelo.__class__.__name__ == 'H2ORandomForestEstimator':
            metricas['rf'] = {
                'variable_importance': modelo.varimp(use_pandas=True),
                'confusion_matrix': perf.confusion_matrix().as_data_frame() if hasattr(perf, 'confusion_matrix') else None
            }
        elif modelo.__class__.__name__ == 'H2ODeepLearningEstimator':
            metricas['dl'] = {
                'scoring_history': modelo.scoring_history(),
                'activation': modelo.activation,
                'hidden_layers': modelo.hidden
            }
            
        return metricas

    def entrenar(self, datos):
        """Entrena usando todas las capacidades de H2O automáticamente"""
        try:
            self.logger.info(f"Iniciando entrenamiento con {datos.shape[0]} registros")
            # 1. Iniciar H2O
            servidor_iniciado = iniciar_servidor_h2o()
            if not servidor_iniciado:
                return pd.DataFrame({'Error': ['No se pudo iniciar H2O']})
            
            # 2. Convertir datos - H2O detecta todo
            h2o_frame = h2o.H2OFrame(datos)
            
            # 3. AutoML con todas las capacidades
            aml = h2o.automl.H2OAutoML(**self.configuracion)
            aml.train(training_frame=h2o_frame)
            
            # 4. Obtener predicciones y métricas avanzadas
            predicciones = aml.predict(h2o_frame)
            predicciones_df = h2o.as_list(predicciones)
            
            # 5. Obtener métricas específicas del mejor modelo
            metricas_modelo = self.obtener_metricas_modelo(aml.leader, h2o_frame)
            
            # 6. Analizar resultados
            reporte = analizar_resultados(
                modelo=aml.leader,
                datos=datos,
                predicciones=predicciones_df,
                tipo_modelo='automl'
            )
            
            # 7. Guardar modelo con todas las métricas
            self.modelo_manager.guardar_modelo(
                modelo=aml.leader,
                tipo_modelo='automl',
                metricas={
                    **reporte['metricas_basicas'],
                    'h2o_metrics': metricas_modelo,
                    'leaderboard': aml.leaderboard.as_data_frame().to_dict(),
                    'training_time': aml.leader.train_time_ms,
                    'model_type': aml.leader.__class__.__name__,
                    'cross_validation': {
                        'metrics': aml.leader.cross_validation_metrics().metrics,
                        'predictions': aml.leader.cross_validation_holdout_predictions().as_data_frame().to_dict() if hasattr(aml.leader, 'cross_validation_holdout_predictions') else None
                    }
                }
            )
            
            # 8. Preparar resultado enriquecido
            df_resultado = pd.DataFrame(predicciones_df)
            df_resultado.columns = ['prediccion']
            
            # Agregar intervalos de confianza si están disponibles
            if hasattr(predicciones, 'predict_intervals'):
                intervalos = predicciones.predict_intervals()
                df_resultado['intervalo_inferior'] = intervalos['lower']
                df_resultado['intervalo_superior'] = intervalos['upper']
            
            return df_resultado
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {str(e)}")
            return pd.DataFrame({'Error': [str(e)]})
        finally:
            self.logger.info("Finalizando entrenamiento")
            detener_servidor() 