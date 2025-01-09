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
        """Obtiene métricas específicas según el tipo de modelo"""
        metricas = {}
        
        try:
            # Obtener performance
            perf = modelo.model_performance(datos)
            
            # Métricas básicas para todos los modelos
            metricas['basic'] = {
                'rmse': perf.rmse() if hasattr(perf, 'rmse') else None,
                'mse': perf.mse() if hasattr(perf, 'mse') else None,
                'r2': perf.r2() if hasattr(perf, 'r2') else None
            }
            
            # Métricas específicas para clasificación
            if modelo._model_json['output']['model_category'] == 'Binomial':
                metricas['clasificacion'] = {
                    'auc': perf.auc() if hasattr(perf, 'auc') else None,
                    'precision': perf.precision() if hasattr(perf, 'precision') else None,
                    'recall': perf.recall() if hasattr(perf, 'recall') else None,
                    'f1': perf.F1() if hasattr(perf, 'F1') else None,
                    'confusion_matrix': perf.confusion_matrix().as_data_frame().to_dict() if hasattr(perf, 'confusion_matrix') else None
                }
            # Métricas específicas para regresión
            else:
                metricas['regresion'] = {
                    'mae': perf.mae() if hasattr(perf, 'mae') else None,
                    'rmsle': perf.rmsle() if hasattr(perf, 'rmsle') else None,
                    'mean_residual_deviance': perf.mean_residual_deviance() if hasattr(perf, 'mean_residual_deviance') else None
                }
            
            return metricas
            
        except Exception as e:
            self.logger.error(f"Error obteniendo métricas: {str(e)}")
            return metricas

    def particionar_datos(self, h2o_frame, objetivo, train_ratio=0.7, valid_ratio=0.15):
        """
        Particiona los datos en conjuntos de entrenamiento, validación y prueba.
        
        Args:
            h2o_frame: H2OFrame con los datos
            objetivo: Variable objetivo
            train_ratio: Proporción para entrenamiento (default 0.7)
            valid_ratio: Proporción para validación (default 0.15)
            
        Returns:
            tuple: (train, valid, test) H2OFrames
        """
        # Dividir primero en train y temp
        train, temp = h2o_frame.split_frame(ratios=[train_ratio], seed=self.configuracion['seed'])
        
        # Dividir temp en validation y test
        valid_ratio_adjusted = valid_ratio / (1 - train_ratio)
        valid, test = temp.split_frame(ratios=[valid_ratio_adjusted], seed=self.configuracion['seed'])
        
        return train, valid, test

    def preprocesar_datos(self, datos, objetivo):
        """
        Preprocesa los datos antes del entrenamiento.
        
        Args:
            datos: DataFrame con los datos
            objetivo: Variable objetivo
            
        Returns:
            H2OFrame: Datos preprocesados
        """
        try:
            self.logger.info("Iniciando preprocesamiento de datos")
            
            # 1. Convertir a H2O Frame
            h2o_frame = h2o.H2OFrame(datos)
            
            # 2. Identificar tipos de columnas
            columnas_numericas = [col for col in h2o_frame.columns if col != objetivo and 
                                h2o_frame[col].type in ['int', 'real']]
            columnas_categoricas = [col for col in h2o_frame.columns if col != objetivo and 
                                  h2o_frame[col].type in ['enum', 'string']]
            
            # 3. Manejo de valores nulos
            for col in h2o_frame.columns:
                if h2o_frame[col].isfactor():
                    h2o_frame[col] = h2o_frame[col].fillna(mode=h2o_frame[col])
                else:
                    h2o_frame[col] = h2o_frame[col].fillna(method="mean")
            
            # 4. Encoding automático para categóricas
            for col in columnas_categoricas:
                h2o_frame[col] = h2o_frame[col].asfactor()
            
            # 5. Normalización de numéricas
            for col in columnas_numericas:
                mean = h2o_frame[col].mean()
                std = h2o_frame[col].sd()
                if std > 0:
                    h2o_frame[col] = (h2o_frame[col] - mean) / std
            
            self.logger.info("Preprocesamiento completado exitosamente")
            return h2o_frame
            
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {str(e)}")
            raise

    def entrenar(self, datos, objetivo):
        """
        Entrena usando todas las capacidades de H2O automáticamente con particionamiento de datos.
        
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
                    'sort_metric': 'AUC',
                    'nfolds': 5  # Agregar validación cruzada
                })
            else:
                self.configuracion.update({
                    'stopping_metric': 'RMSE',
                    'sort_metric': 'RMSE',
                    'nfolds': 5  # Agregar validación cruzada
                })

            # 4. Preprocesar datos
            h2o_frame = self.preprocesar_datos(datos, objetivo)
            
            # Preparar objetivo para clasificación
            if tipo_modelo == 'clasificacion':
                h2o_frame[objetivo] = h2o_frame[objetivo].asfactor()
            
            # 5. Particionar datos
            train, valid, test = self.particionar_datos(h2o_frame, objetivo)
            
            # 6. AutoML con todas las capacidades y validación cruzada
            aml = h2o.automl.H2OAutoML(
                max_models=self.configuracion['max_models'],
                seed=self.configuracion['seed'],
                max_runtime_secs=self.configuracion['max_runtime_secs'],
                include_algos=self.configuracion['include_algos'],
                nfolds=self.configuracion['nfolds'],
                keep_cross_validation_predictions=True
            )
            
            # Entrenar con conjuntos de validación
            aml.train(
                y=objetivo,
                training_frame=train,
                validation_frame=valid,
                leaderboard_frame=test
            )
            
            # 7. Obtener predicciones del mejor modelo en conjunto de prueba
            predicciones = aml.leader.predict(test)
            predicciones_df = predicciones.as_data_frame()
            
            # 8. Obtener métricas del mejor modelo en conjunto de prueba
            metricas = self.obtener_metricas_modelo(aml.leader, test)
            
            # 9. Guardar solo el mejor modelo con nombre descriptivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            nombre_modelo = f"{tipo_modelo}_{objetivo}_{aml.leader.__class__.__name__}_{timestamp}"
            
            self.modelo_manager.guardar_modelo(
                modelo=aml.leader,
                nombre=nombre_modelo,
                metricas=metricas
            )
            
            # 10. Preparar resultado
            return {
                'predicciones': predicciones_df,
                'tipo_modelo': tipo_modelo,
                'metricas': metricas,
                'modelo_info': {
                    'nombre': nombre_modelo,
                    'tipo': aml.leader.__class__.__name__,
                    'variables_importantes': aml.leader.varimp(use_pandas=True) if hasattr(aml.leader, 'varimp') else None,
                    'leaderboard': aml.leaderboard.as_data_frame().to_dict(),
                    'tiempo_total': aml.training_time_ms if hasattr(aml, 'training_time_ms') else None,
                    'particion_datos': {
                        'train_size': train.shape[0],
                        'valid_size': valid.shape[0],
                        'test_size': test.shape[0]
                    }
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

    def analizar_dataset(self, datos, objetivo):
        """
        Realiza un análisis completo del dataset antes del modelado.
        
        Args:
            datos: DataFrame con los datos
            objetivo: Variable objetivo
            
        Returns:
            dict: Análisis completo del dataset
        """
        try:
            self.logger.info(f"Analizando dataset para objetivo: {objetivo}")
            
            # 1. Estadísticas básicas
            stats = {
                'n_registros': len(datos),
                'n_features': len(datos.columns),
                'memoria_uso': datos.memory_usage().sum() / 1024**2,  # MB
                'valores_nulos': datos.isnull().sum().to_dict(),
                'tipos_datos': datos.dtypes.astype(str).to_dict()
            }
            
            # 2. Análisis de la variable objetivo
            target_stats = {
                'tipo': self.determinar_tipo_modelo(datos, objetivo),
                'distribucion': datos[objetivo].value_counts().to_dict() if datos[objetivo].dtype == 'object' 
                               else {'mean': datos[objetivo].mean(), 'std': datos[objetivo].std(),
                                    'min': datos[objetivo].min(), 'max': datos[objetivo].max()}
            }
            
            # 3. Análisis de correlaciones
            correlaciones = {}
            numericas = datos.select_dtypes(include=['int64', 'float64']).columns
            if len(numericas) > 0:
                corr_matrix = datos[numericas].corr()
                # Encontrar correlaciones importantes con objetivo
                if objetivo in numericas:
                    correlaciones = {col: corr_matrix[objetivo][col] 
                                   for col in numericas if abs(corr_matrix[objetivo][col]) > 0.5 
                                   and col != objetivo}
            
            # 4. Detección de desbalanceo (para clasificación)
            desbalanceo = None
            if target_stats['tipo'] == 'clasificacion':
                valores = datos[objetivo].value_counts()
                ratio = valores.min() / valores.max()
                desbalanceo = {
                    'ratio_desbalanceo': ratio,
                    'distribucion_clases': valores.to_dict(),
                    'necesita_balanceo': ratio < 0.2  # umbral de 20%
                }
            
            return {
                'estadisticas_basicas': stats,
                'analisis_objetivo': target_stats,
                'correlaciones_importantes': correlaciones,
                'analisis_desbalanceo': desbalanceo,
                'recomendaciones': self._generar_recomendaciones(stats, target_stats, correlaciones, desbalanceo)
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de dataset: {str(e)}")
            raise

    def _generar_recomendaciones(self, stats, target_stats, correlaciones, desbalanceo):
        """Genera recomendaciones basadas en el análisis del dataset"""
        recomendaciones = []
        
        # Análisis de valores nulos
        nulos_importantes = {k: v for k, v in stats['valores_nulos'].items() if v > 0}
        if nulos_importantes:
            recomendaciones.append({
                'tipo': 'valores_nulos',
                'descripcion': f"Se encontraron valores nulos en {len(nulos_importantes)} columnas",
                'sugerencia': "Considerar estrategias específicas de imputación"
            })
        
        # Análisis de correlaciones
        if correlaciones:
            recomendaciones.append({
                'tipo': 'correlaciones',
                'descripcion': f"Se encontraron {len(correlaciones)} variables altamente correlacionadas",
                'sugerencia': "Considerar reducción de dimensionalidad o selección de features"
            })
        
        # Análisis de desbalanceo
        if desbalanceo and desbalanceo['necesita_balanceo']:
            recomendaciones.append({
                'tipo': 'desbalanceo',
                'descripcion': f"Dataset desbalanceado (ratio: {desbalanceo['ratio_desbalanceo']:.2f})",
                'sugerencia': "Considerar técnicas de balanceo como SMOTE o class_weight"
            })
        
        return recomendaciones

    def interpretar_modelo(self, modelo, datos_test, objetivo):
        """
        Realiza una interpretación detallada del modelo.
        
        Args:
            modelo: Modelo H2O entrenado
            datos_test: Datos de prueba
            objetivo: Variable objetivo
            
        Returns:
            dict: Interpretación detallada del modelo
        """
        try:
            self.logger.info("Iniciando interpretación del modelo")
            
            interpretacion = {
                'tipo_modelo': modelo.__class__.__name__,
                'importancia_variables': None,
                'performance': None,
                'analisis_residuos': None,
                'interpretabilidad_local': None
            }
            
            # 1. Importancia de variables
            if hasattr(modelo, 'varimp'):
                interpretacion['importancia_variables'] = modelo.varimp(use_pandas=True)
            
            # 2. Performance detallado
            perf = modelo.model_performance(datos_test)
            interpretacion['performance'] = self.obtener_metricas_modelo(modelo, datos_test)
            
            # 3. Análisis de residuos (para regresión)
            if modelo._model_json['output']['model_category'] == 'Regression':
                predicciones = modelo.predict(datos_test)
                residuos = predicciones.as_data_frame()['predict'] - datos_test[objetivo].as_data_frame()
                interpretacion['analisis_residuos'] = {
                    'mean': float(residuos.mean()),
                    'std': float(residuos.std()),
                    'max_abs_error': float(abs(residuos).max())
                }
            
            # 4. Interpretabilidad local (LIME o SHAP si está disponible)
            try:
                if hasattr(modelo, 'predict_contributions'):
                    muestra = datos_test[:100]  # Analizar primeras 100 muestras
                    contribuciones = modelo.predict_contributions(muestra)
                    interpretacion['interpretabilidad_local'] = contribuciones.as_data_frame().mean().to_dict()
            except:
                self.logger.warning("No se pudo realizar interpretación local del modelo")
            
            return interpretacion
            
        except Exception as e:
            self.logger.error(f"Error en interpretación del modelo: {str(e)}")
            raise

    def comparar_modelos(self, aml):
        """
        Compara los modelos generados por AutoML.
        
        Args:
            aml: Objeto H2O AutoML entrenado
            
        Returns:
            dict: Comparación detallada de modelos
        """
        try:
            self.logger.info("Comparando modelos generados")
            
            # 1. Obtener leaderboard detallado
            leaderboard = aml.leaderboard.as_data_frame()
            
            # 2. Analizar diversidad de modelos
            tipos_modelos = leaderboard.index.str.split('_').str[0].value_counts().to_dict()
            
            # 3. Análisis de trade-offs
            mejor_modelo = aml.leader
            modelos_pareto = []  # Modelos en la frontera de Pareto
            
            comparacion = {
                'n_modelos_total': len(leaderboard),
                'tipos_modelos': tipos_modelos,
                'mejor_modelo': {
                    'nombre': mejor_modelo.__class__.__name__,
                    'metricas': self.obtener_metricas_modelo(mejor_modelo, aml.leaderboard_frame),
                    'tiempo_entrenamiento': mejor_modelo.training_time_ms if hasattr(mejor_modelo, 'training_time_ms') else None
                },
                'distribucion_performance': {
                    'mean': float(leaderboard[aml.sort_metric].mean()),
                    'std': float(leaderboard[aml.sort_metric].std()),
                    'range': float(leaderboard[aml.sort_metric].max() - leaderboard[aml.sort_metric].min())
                }
            }
            
            return comparacion
            
        except Exception as e:
            self.logger.error(f"Error en comparación de modelos: {str(e)}")
            raise 