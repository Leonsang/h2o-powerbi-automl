import pandas as pd
import h2o
import os
from datetime import datetime
from .modelo_manager import ModeloManager
from .init_h2o_server import iniciar_servidor_h2o, detener_servidor
from .analizar_resultados import analizar_resultados
from .logger import Logger
import numpy as np

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
        self.logger = Logger('h2o_modelo')
        self.modelo_manager = ModeloManager()
        self.configuracion = {
            'max_models': 10,
            'seed': 42,
            'max_runtime_secs': 300,
            'include_algos': ['GBM', 'RF', 'DeepLearning', 'GLM']
        }

    def obtener_columnas_posibles(self, datos):
        """
        Analiza el dataset y determina qué columnas son adecuadas como objetivo
        
        Args:
            datos: DataFrame con los datos
            
        Returns:
            list: Lista de columnas que pueden usarse como objetivo
        """
        columnas_posibles = []
        
        for columna in datos.columns:
            # Verificar tipo de datos
            dtype = datos[columna].dtype
            n_unicos = datos[columna].nunique()
            
            # Criterios más flexibles:
            if (
                # Numéricas
                np.issubdtype(dtype, np.number) or
                # Categóricas 
                dtype == 'object' or
                # Booleanas
                dtype == 'bool' or
                # Binarias numéricas (0/1)
                (np.issubdtype(dtype, np.number) and set(datos[columna].unique()) <= {0, 1})
            ):
                columnas_posibles.append(columna)
                
        return columnas_posibles

    def determinar_tipo_modelo(self, datos, columna):
        """
        Determina el tipo de modelo adecuado para la columna objetivo
        
        Returns:
            str: 'regresion' o 'clasificacion'
        """
        dtype = datos[columna].dtype
        n_unicos = datos[columna].nunique()
        
        if np.issubdtype(dtype, np.number) and n_unicos > 10:
            return 'regresion'
        else:
            return 'clasificacion'

    def obtener_metricas_modelo(self, modelo, datos):
        """Obtiene métricas básicas del modelo"""
        metricas = {}
        
        # Métricas básicas
        perf = modelo.model_performance(datos)
        metricas['basic'] = {
            'rmse': perf.rmse() if hasattr(perf, 'rmse') else None,
            'mse': perf.mse() if hasattr(perf, 'mse') else None,
            'r2': perf.r2() if hasattr(perf, 'r2') else None
        }
        
        return metricas

    def entrenar(self, datos, objetivo):
        """
        Entrena usando todas las capacidades de H2O automáticamente.
        
        Args:
            datos: DataFrame con los datos
            objetivo: Variable objetivo
            
        Returns:
            dict: Resultados del entrenamiento
        """
        try:
            # 1. Validar columna objetivo
            if objetivo not in self.obtener_columnas_posibles(datos):
                raise ValueError(f"Columna {objetivo} no es adecuada como objetivo")

            # 2. Determinar tipo de modelo
            tipo_modelo = self.determinar_tipo_modelo(datos, objetivo)
            
            # 3. Configurar AutoML según tipo
            if tipo_modelo == 'clasificacion':
                self.configuracion.update({
                    'stopping_metric': 'AUC',
                    'sort_metric': 'AUC'
                })
            else:
                self.configuracion.update({
                    'stopping_metric': 'RMSE',
                    'sort_metric': 'RMSE'
                })

            # 4. Convertir datos a H2O y preparar columna objetivo
            h2o_frame = h2o.H2OFrame(datos)
            
            # Determinar tipo de modelo y preparar objetivo
            tipo_modelo = self.determinar_tipo_modelo(datos, objetivo)
            if tipo_modelo == 'clasificacion':
                h2o_frame[objetivo] = h2o_frame[objetivo].asfactor()
            
            # 5. AutoML con todas las capacidades
            aml = h2o.automl.H2OAutoML(
                max_models=10,
                seed=42,
                max_runtime_secs=300
            )
            aml.train(y=objetivo, training_frame=h2o_frame)
            
            # 6. Obtener predicciones del mejor modelo
            predicciones = aml.leader.predict(h2o_frame)
            predicciones_df = predicciones.as_data_frame()
            
            # 7. Obtener métricas del mejor modelo
            metricas = self.obtener_metricas_modelo(aml.leader, h2o_frame)
            
            # 8. Guardar solo el mejor modelo con nombre descriptivo
            tipo_modelo = self.determinar_tipo_modelo(datos, objetivo)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            nombre_modelo = f"{tipo_modelo}_{objetivo}_{aml.leader.__class__.__name__}_{timestamp}"
            
            self.modelo_manager.guardar_modelo(
                modelo=aml.leader,
                nombre=nombre_modelo,
                metricas=metricas
            )
            
            # 9. Preparar resultado
            return {
                'predicciones': predicciones_df,
                'tipo_modelo': tipo_modelo,
                'metricas': metricas,
                'modelo_info': {
                    'nombre': nombre_modelo,
                    'tipo': aml.leader.__class__.__name__,
                    'variables_importantes': aml.leader.varimp(use_pandas=True) if hasattr(aml.leader, 'varimp') else None,
                    'leaderboard': aml.leaderboard.as_data_frame().to_dict(),
                    'tiempo_total': aml.training_time_ms if hasattr(aml, 'training_time_ms') else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {str(e)}")
            raise

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