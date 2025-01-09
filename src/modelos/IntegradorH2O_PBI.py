import pandas as pd
import h2o
import os
from datetime import datetime
from .modelo_manager import ModeloManager
from .init_h2o_server import iniciar_servidor_h2o, detener_servidor
from .analizar_resultados import analizar_resultados
from .logger import Logger
from .feature_engineering import FeatureEngineering
from .visualizaciones import Visualizador
from .mlops import MLOpsManager
from .asistente_ia import AsistenteIA
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
        """Inicializa H2O con todas sus capacidades"""
        self.logger = Logger('h2o_modelo')
        self.modelo_manager = ModeloManager()
        self.feature_engineering = FeatureEngineering()
        self.visualizador = Visualizador()
        self.mlops_manager = MLOpsManager()
        self.asistente = AsistenteIA()
        self.base_dir = os.path.abspath(os.getcwd())
        self.configuracion = {
            'max_models': 10,
            'seed': 42,
            'max_runtime_secs': 300,
            'include_algos': ["DRF", "GBM", "XGBoost", "GLM"],
            'feature_engineering': {
                'detectar_outliers': True,
                'crear_interacciones': True,
                'reducir_dimensionalidad': False,
                'seleccionar_features': True,
                'k_mejores_features': 10
            },
            'monitoreo': {
                'activar_mlops': True,
                'frecuencia_monitoreo': '1d',
                'umbral_drift': 0.1
            },
            'visualizaciones': {
                'generar_graficos': True,
                'guardar_graficos': True
            }
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
                    'confusion_matrix': perf.confusion_matrix().table.as_data_frame().to_dict() if hasattr(perf, 'confusion_matrix') else None
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
            
            # Convertir a H2O Frame
            h2o_frame = h2o.H2OFrame(datos)
            
            # Identificar tipos de columnas
            columnas_numericas = [col for col in h2o_frame.columns if col != objetivo and 
                                h2o_frame[col].type in ['int', 'real']]
            columnas_categoricas = [col for col in h2o_frame.columns if col != objetivo and 
                                  h2o_frame[col].type in ['enum', 'string']]
            
            # Encoding automático para categóricas
            for col in columnas_categoricas:
                h2o_frame[col] = h2o_frame[col].asfactor()
            
            # Normalización de numéricas
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

    def procesar_features(self, datos, objetivo=None):
        """Aplica feature engineering según la configuración"""
        try:
            self.logger.info("Iniciando procesamiento de features")
            config_fe = self.configuracion['feature_engineering']
            
            # 1. Detección y procesamiento de outliers
            if config_fe['detectar_outliers']:
                datos = self.feature_engineering.outlier_detector.process(datos)
            
            # 2. Crear features de interacción
            if config_fe['crear_interacciones']:
                columnas_numericas = datos.select_dtypes(include=['int64', 'float64']).columns
                datos = self.feature_engineering.crear_features_interaccion(datos, columnas_numericas)
            
            # 3. Selección de features
            if config_fe['seleccionar_features'] and objetivo:
                datos = self.feature_engineering.seleccionar_features(
                    datos, 
                    objetivo,
                    config_fe['k_mejores_features']
                )
            
            # 4. Reducción de dimensionalidad
            if config_fe['reducir_dimensionalidad']:
                datos = self.feature_engineering.reducir_dimensionalidad(datos)
            
            return datos
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento de features: {str(e)}")
            raise

    def entrenar(self, datos, objetivo):
        """Proceso completo de entrenamiento con todas las capacidades"""
        try:
            # 1. Validar columna objetivo
            if objetivo not in self.obtener_columnas_posibles(datos):
                raise ValueError(f"Columna {objetivo} no es adecuada como objetivo")

            # 2. Feature Engineering
            self.logger.info("Aplicando feature engineering")
            datos_procesados = self.procesar_features(datos, objetivo)
            reporte_features = self.feature_engineering.generar_reporte_features(datos, datos_procesados)

            # 3. Determinar tipo de modelo
            tipo_modelo = self.determinar_tipo_modelo(datos_procesados, objetivo)
            
            # 4. Configurar AutoML según tipo
            self._configurar_automl(tipo_modelo)

            # 5. Preprocesar datos y convertir a H2O Frame
            h2o_frame = h2o.H2OFrame(datos_procesados)
            
            # Convertir objetivo a factor si es clasificación
            if tipo_modelo == 'clasificacion':
                h2o_frame[objetivo] = h2o_frame[objetivo].asfactor()
            
            # 6. Particionar datos
            train, valid, test = self.particionar_datos(h2o_frame, objetivo)
            
            # 7. Entrenar modelo
            aml = h2o.automl.H2OAutoML(
                max_models=self.configuracion['max_models'],
                seed=self.configuracion['seed'],
                max_runtime_secs=self.configuracion['max_runtime_secs'],
                include_algos=self.configuracion['include_algos'],
                nfolds=self.configuracion.get('nfolds', 5),
                keep_cross_validation_predictions=True,
                sort_metric=self.configuracion.get('sort_metric', 'AUTO'),
                stopping_metric=self.configuracion.get('stopping_metric', 'AUTO'),
                verbosity='info'
            )
            
            aml.train(
                y=objetivo,
                training_frame=train,
                validation_frame=valid,
                leaderboard_frame=test
            )
            
            # 8. Generar predicciones y métricas
            predicciones = aml.leader.predict(test)
            metricas = self.obtener_metricas_modelo(aml.leader, test)
            
            # 9. Preparar resultados
            resultados = {
                'modelo': aml.leader,
                'metricas': metricas,
                'predicciones': predicciones.as_data_frame(),
                'datos_test': test,
                'importancia_variables': self._obtener_importancia_variables(aml.leader),
                'leaderboard': aml.leaderboard.as_data_frame(),
                'tipo_modelo': tipo_modelo,
                'configuracion': self.configuracion
            }
            
            return resultados
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {str(e)}")
            raise

    def _configurar_automl(self, tipo_modelo):
        """Configura AutoML según el tipo de modelo"""
        if tipo_modelo == 'clasificacion':
            self.configuracion.update({
                'stopping_metric': 'AUC',
                'sort_metric': 'AUC',
                'nfolds': 5
            })
        else:
            self.configuracion.update({
                'stopping_metric': 'RMSE',
                'sort_metric': 'RMSE',
                'nfolds': 5
            })

    def _configurar_mlops(self, modelo, datos, objetivo):
        """Configura el monitoreo MLOps"""
        try:
            metadata = {
                'tipo_modelo': self.determinar_tipo_modelo(datos, objetivo),
                'features': datos.columns.tolist(),
                'configuracion_feature_engineering': self.configuracion['feature_engineering'],
                'configuracion_modelo': self.configuracion
            }
            
            modelo_id = self.mlops_manager.registrar_modelo(modelo, metadata)
            
            # Configurar monitoreo
            self.mlops_manager.monitorear_modelo(
                modelo_id,
                datos,
                objetivo
            )
            
        except Exception as e:
            self.logger.error(f"Error configurando MLOps: {str(e)}")
            raise

    def _obtener_importancia_variables(self, modelo):
        """Obtiene la importancia de variables del modelo"""
        try:
            if hasattr(modelo, 'varimp'):
                importancia = modelo.varimp(use_pandas=True)
                return dict(zip(importancia['variable'], importancia['relative_importance']))
            return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo importancia de variables: {str(e)}")
            return {}

    def _crear_estructura_modelo(self, modelo_id):
        """Crea la estructura de directorios para un modelo"""
        estructura = {
            'modelo': os.path.join(self.base_dir, 'mlops', 'modelos', modelo_id),
            'metricas': os.path.join(self.base_dir, 'mlops', 'metricas', modelo_id),
            'visualizaciones': os.path.join(self.base_dir, 'output', 'visualizaciones', modelo_id),
            'reportes': os.path.join(self.base_dir, 'output', 'reportes', modelo_id),
            'logs': os.path.join(self.base_dir, 'logs', modelo_id)
        }
        
        for directorio in estructura.values():
            os.makedirs(directorio, exist_ok=True)
            
        return estructura

    def ejecutar_flujo_completo(self, datos, objetivo):
        """Ejecuta el flujo completo de análisis, modelado, visualización e interpretación."""
        try:
            self.logger.info("Iniciando flujo completo de análisis y modelado")
            
            # Generar ID del modelo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            modelo_id = f"modelo_{timestamp}"
            
            # Crear estructura de directorios
            estructura = self._crear_estructura_modelo(modelo_id)
            
            # 1. Análisis inicial del dataset
            analisis_dataset = self.analizar_dataset(datos, objetivo)
            self.logger.info("Análisis de dataset completado")
            
            # 2. Entrenamiento del modelo
            resultados_modelo = self.entrenar(datos, objetivo)
            resultados_modelo['modelo_id'] = modelo_id
            self.logger.info("Entrenamiento completado")
            
            # 3. Generar visualizaciones
            self.logger.info("Generando visualizaciones")
            visualizaciones = {}
            
            # Generar y guardar visualizaciones
            figs = {
                'distribucion': self.visualizador.plot_distribucion_variables(datos),
                'correlaciones': self.visualizador.plot_correlaciones(datos),
                'importancia': self.visualizador.plot_importancia_variables(
                    resultados_modelo['importancia_variables']
                )
            }
            
            # Guardar cada visualización
            for nombre, fig in figs.items():
                if fig is not None:
                    ruta = self.visualizador.guardar_visualizacion(fig, nombre, modelo_id)
                    visualizaciones[nombre] = ruta
            
            # 4. Generar interpretación IA
            self.logger.info("Generando interpretación IA")
            interpretacion = self.asistente.generar_reporte_completo({
                'analisis_dataset': analisis_dataset,
                'metricas': resultados_modelo['metricas'],
                'interpretacion': {
                    'importancia_variables': resultados_modelo['importancia_variables']
                },
                'objetivo': objetivo
            })
            
            # Guardar reporte IA
            ruta_reporte = os.path.join(estructura['reportes'], 'reporte_ia.txt')
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write(interpretacion)
            
            # 5. Configurar MLOps
            if self.configuracion['monitoreo']['activar_mlops']:
                self.logger.info("Configurando MLOps")
                self._configurar_mlops(
                    resultados_modelo['modelo'],
                    datos,
                    objetivo
                )
            
            # 6. Integrar todos los resultados
            resultados_completos = {
                'analisis_dataset': analisis_dataset,
                'modelo': resultados_modelo,
                'visualizaciones': visualizaciones,
                'interpretacion_ia': interpretacion,
                'configuracion': self.configuracion,
                'timestamp': timestamp,
                'modelo_id': modelo_id,
                'rutas': estructura
            }
            
            self.logger.info("Flujo completo ejecutado exitosamente")
            return resultados_completos
            
        except Exception as e:
            self.logger.error(f"Error en flujo completo: {str(e)}")
            raise

    def generar_reporte_power_bi(self, resultados):
        """
        Genera un reporte estructurado para Power BI.
        
        Args:
            resultados: Dict con los resultados del flujo completo
            
        Returns:
            dict: Reporte estructurado para Power BI
        """
        try:
            self.logger.info("Generando reporte para Power BI")
            
            # 1. Métricas principales
            metricas_principales = pd.DataFrame({
                'Metrica': ['AUC', 'Precisión', 'Recall'],
                'Valor': [
                    resultados['modelo']['metricas']['clasificacion']['auc'],
                    resultados['modelo']['metricas']['clasificacion']['precision'][0][1],
                    resultados['modelo']['metricas']['clasificacion']['recall'][0][1]
                ]
            })
            
            # 2. Importancia de variables
            importancia_vars = pd.DataFrame([
                {'Variable': k, 'Importancia': v}
                for k, v in resultados['modelo']['importancia_variables'].items()
            ])
            
            # 3. Insights principales
            insights = pd.DataFrame([
                {'Categoria': 'Dataset', 'Insight': resultados['analisis_dataset']['recomendaciones'][0]['descripcion']},
                {'Categoria': 'Modelo', 'Insight': resultados['interpretacion_ia'].split('\n')[2]},
                {'Categoria': 'Recomendaciones', 'Insight': resultados['interpretacion_ia'].split('\n')[-2]}
            ])
            
            return {
                'metricas': metricas_principales,
                'importancia_variables': importancia_vars,
                'insights': insights,
                'visualizaciones': resultados['visualizaciones']
            }
            
        except Exception as e:
            self.logger.error(f"Error generando reporte Power BI: {str(e)}")
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