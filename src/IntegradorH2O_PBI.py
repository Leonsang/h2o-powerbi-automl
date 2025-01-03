import pandas as pd
import h2o
import os
from datetime import datetime
from .modelo_manager import ModeloManager
from .init_h2o_server import iniciar_servidor_h2o, detener_servidor
from .analizar_resultados import analizar_resultados
from .logger import Logger

class H2OModeloAvanzado:
    """
    Integrador de H2O AutoML con Power BI para democratización del análisis predictivo.
    
    Este módulo busca:
    1. Facilitar el acceso a ML avanzado para usuarios de negocio
    2. Proporcionar explicaciones claras de las predicciones
    3. Fomentar decisiones basadas en datos
    """
    
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
        """
        Obtiene métricas completas según el tipo de modelo.
        
        Args:
            modelo: Modelo H2O entrenado
            datos: Datos de evaluación
            
        Returns:
            Dict con métricas específicas del modelo
        """
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
        """
        Entrena usando todas las capacidades de H2O automáticamente.
        
        Args:
            datos: DataFrame con los datos de entrenamiento
            
        Returns:
            DataFrame con predicciones y métricas
        """
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

    def analizar_causalidad(self, datos, objetivo, variables):
        """
        Analiza relaciones causales entre variables.
        
        Args:
            datos: DataFrame con datos históricos
            objetivo: Variable a explicar
            variables: Variables explicativas
            
        Returns:
            Dict con:
            - relaciones_causales: Matriz de causalidad
            - importancia: Ranking de variables
            - recomendaciones: Acciones sugeridas
        """
        try:
            # Validaciones de entrada
            if not isinstance(datos, pd.DataFrame):
                raise ValueError("datos debe ser un DataFrame")
            if objetivo not in datos.columns:
                raise ValueError(f"objetivo {objetivo} no encontrado en datos")
            if not all(var in datos.columns for var in variables):
                raise ValueError("algunas variables no encontradas en datos")
            
            self.logger.info(f"Analizando causalidad para {objetivo}")
            
            # 1. Convertir a H2O Frame
            h2o_frame = h2o.H2OFrame(datos)
            
            # 2. Entrenar modelo ligero para importancia
            modelo_rapido = h2o.estimators.gbm.H2OGradientBoostingEstimator(
                ntrees=10,
                max_depth=3,
                learn_rate=0.1
            )
            modelo_rapido.train(x=variables, y=objetivo, training_frame=h2o_frame)
            
            # 3. Obtener importancia de variables
            importancia = modelo_rapido.varimp(use_pandas=True)
            
            # 4. Calcular correlaciones
            correlaciones = datos[variables + [objetivo]].corr()
            
            # 5. Generar recomendaciones
            recomendaciones = []
            for var, imp in zip(importancia['variable'], importancia['relative_importance']):
                if imp > 0.1:  # Variables importantes
                    corr = correlaciones.loc[var, objetivo]
                    if abs(corr) > 0.5:
                        direccion = "aumenta" if corr > 0 else "disminuye"
                        recomendaciones.append(
                            f"Cuando {var} aumenta, {objetivo} {direccion} significativamente"
                        )
            
            return {
                'relaciones_causales': correlaciones.to_dict(),
                'importancia': importancia.to_dict(),
                'recomendaciones': recomendaciones,
                'metricas_modelo': {
                    'r2': modelo_rapido.r2(),
                    'rmse': modelo_rapido.rmse()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis causal: {str(e)}")
            return {'error': str(e)}

    def explicar_prediccion(self, prediccion_id, nivel_detalle='basico'):
        """
        Explica en lenguaje natural el por qué de una predicción.
        
        Args:
            prediccion_id: ID de la predicción a explicar
            nivel_detalle: 'basico' o 'detallado'
            
        Returns:
            Dict con:
            - explicacion: Texto en lenguaje natural
            - factores: Variables más influyentes
            - confianza: Nivel de certeza
        """
        try:
            self.logger.info(f"Explicando predicción {prediccion_id}")
            
            # 1. Obtener predicción y modelo
            prediccion = self.modelo_manager.obtener_prediccion(prediccion_id)
            modelo = self.modelo_manager.obtener_ultimo_modelo()
            
            if not prediccion or not modelo:
                return {'error': 'Predicción o modelo no encontrado'}
            
            # 2. Obtener contribuciones de variables
            contribuciones = modelo.predict_contributions(prediccion)
            
            # 3. Ordenar factores por importancia
            factores = []
            for variable, contribucion in contribuciones.items():
                impacto = "aumenta" if contribucion > 0 else "disminuye"
                factores.append({
                    'variable': variable,
                    'contribucion': contribucion,
                    'descripcion': f"{variable} {impacto} el resultado en {abs(contribucion):.2f}"
                })
            
            # 4. Calcular confianza
            confianza = modelo.model_performance().r2()
            
            # 5. Generar explicación
            if nivel_detalle == 'basico':
                top_factores = sorted(factores, key=lambda x: abs(x['contribucion']))[:3]
                explicacion = f"La predicción se basa principalmente en: "
                explicacion += ", ".join([f.get('descripcion') for f in top_factores])
            else:
                explicacion = "Análisis detallado de factores:\n"
                for f in sorted(factores, key=lambda x: abs(x['contribucion'])):
                    explicacion += f"- {f['descripcion']}\n"
            
            return {
                'explicacion': explicacion,
                'factores': factores,
                'confianza': confianza,
                'metricas_adicionales': {
                    'num_factores': len(factores),
                    'impacto_total': sum(abs(f['contribucion']) for f in factores)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error explicando predicción: {str(e)}")
            return {'error': str(e)} 