import pandas as pd
import numpy as np
from src.logger import Logger
from src.metricas import MetricasManager
from src.interpretabilidad import InterpretabilidadManager
from src.visualizaciones import VisualizacionesManager
from src.modelo_manager import ModeloManager

logger = Logger('analisis_modelo')

class AnalizadorModelo:
    """Coordina el análisis completo de modelos"""
    
    def __init__(self, output_dir='./output'):
        self.logger = logger
        self.output_dir = output_dir
        self.metricas = MetricasManager()
        self.interpretabilidad = InterpretabilidadManager()
        self.visualizaciones = VisualizacionesManager(f"{output_dir}/graficos")
        self.modelo_manager = ModeloManager()

    def analisis_completo(self, modelo, datos, objetivo, tipo_modelo='auto'):
        """
        Realiza un análisis completo del modelo
        
        Args:
            modelo: Modelo entrenado
            datos: DataFrame con los datos
            objetivo: Variable objetivo
            tipo_modelo: 'clasificacion', 'regresion' o 'auto'
        """
        try:
            self.logger.info("Iniciando análisis completo del modelo")
            
            # Determinar tipo de modelo si es auto
            if tipo_modelo == 'auto':
                tipo_modelo = 'clasificacion' if hasattr(modelo, 'predict_proba') else 'regresion'
            
            # 1. Análisis de datos
            analisis_datos = self.analizar_datos(datos, objetivo)
            
            # 2. Evaluación del modelo
            evaluacion = self.evaluar_modelo(modelo, datos, objetivo, tipo_modelo)
            
            # 3. Interpretabilidad
            interpretacion = self.analizar_interpretabilidad(modelo, datos, objetivo)
            
            # 4. Generar visualizaciones
            visualizaciones = self.generar_visualizaciones(
                modelo, datos, objetivo, 
                evaluacion, interpretacion,
                tipo_modelo
            )
            
            # 5. Generar recomendaciones
            recomendaciones = self.generar_recomendaciones(
                analisis_datos,
                evaluacion,
                interpretacion
            )
            
            return {
                'analisis_datos': analisis_datos,
                'evaluacion': evaluacion,
                'interpretacion': interpretacion,
                'visualizaciones': visualizaciones,
                'recomendaciones': recomendaciones
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis completo: {str(e)}")
            raise

    def analizar_datos(self, datos, objetivo):
        """Analiza el conjunto de datos"""
        try:
            analisis = {
                'estadisticas': self._calcular_estadisticas(datos),
                'distribucion_objetivo': self._analizar_distribucion(datos[objetivo]),
                'correlaciones': self._analizar_correlaciones(datos, objetivo),
                'valores_faltantes': self._analizar_valores_faltantes(datos)
            }
            return analisis
        except Exception as e:
            self.logger.error(f"Error en análisis de datos: {str(e)}")
            raise

    def evaluar_modelo(self, modelo, datos, objetivo, tipo_modelo):
        """Evalúa el rendimiento del modelo"""
        try:
            X = datos.drop(columns=[objetivo])
            y = datos[objetivo]
            
            # Predicciones
            y_pred = modelo.predict(X)
            y_proba = modelo.predict_proba(X) if hasattr(modelo, 'predict_proba') else None
            
            # Métricas según tipo de modelo
            if tipo_modelo == 'clasificacion':
                metricas = self.metricas.calcular_metricas_clasificacion(y, y_pred, y_proba)
            else:
                metricas = self.metricas.calcular_metricas_regresion(y, y_pred)
                metricas['analisis_residuos'] = self.metricas.analizar_residuos(y, y_pred)
            
            # Validación cruzada
            cv_results = self.metricas.evaluar_cross_validation(
                modelo, X, y, tipo_modelo=tipo_modelo
            )
            
            return {
                'metricas': metricas,
                'cross_validation': cv_results,
                'predicciones': {
                    'y_pred': y_pred,
                    'y_proba': y_proba
                }
            }
        except Exception as e:
            self.logger.error(f"Error en evaluación del modelo: {str(e)}")
            raise

    def analizar_interpretabilidad(self, modelo, datos, objetivo):
        """Analiza la interpretabilidad del modelo"""
        try:
            # Análisis global
            analisis_global = self.interpretabilidad.analisis_global(
                modelo, datos, objetivo
            )
            
            # Análisis local para casos interesantes
            indices_interesantes = self._identificar_casos_interesantes(
                modelo, datos, objetivo
            )
            
            analisis_local = self.interpretabilidad.analisis_local(
                modelo, datos, indices_interesantes
            )
            
            return {
                'global': analisis_global,
                'local': analisis_local
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de interpretabilidad: {str(e)}")
            raise

    def generar_visualizaciones(self, modelo, datos, objetivo, evaluacion, 
                              interpretacion, tipo_modelo):
        """Genera todas las visualizaciones necesarias"""
        try:
            visualizaciones = {}
            
            # 1. Visualizaciones de datos
            visualizaciones['distribucion_objetivo'] = self.visualizaciones.visualizar_distribucion_objetivo(
                datos, objetivo, tipo_modelo
            )
            
            visualizaciones['correlaciones'] = self.visualizaciones.visualizar_correlaciones(
                datos, objetivo
            )
            
            # 2. Visualizaciones de rendimiento
            visualizaciones['predicciones'] = self.visualizaciones.visualizar_predicciones_vs_real(
                evaluacion['predicciones']['y_true'],
                evaluacion['predicciones']['y_pred'],
                tipo_modelo
            )
            
            if tipo_modelo == 'regresion':
                visualizaciones['residuos'] = self.visualizaciones.visualizar_residuos(
                    evaluacion['metricas']['analisis_residuos']['residuos']
                )
            
            # 3. Visualizaciones de interpretabilidad
            visualizaciones['importancia_variables'] = self.visualizaciones.visualizar_importancia_variables(
                interpretacion['global']['importancia_variables']
            )
            
            visualizaciones['shap_summary'] = self.visualizaciones.visualizar_shap_summary(
                interpretacion['global']['shap_values']['values'],
                datos.columns
            )
            
            # Guardar visualizaciones
            rutas = {}
            for nombre, fig in visualizaciones.items():
                ruta = self.visualizaciones.guardar_visualizacion(fig, nombre)
                rutas[nombre] = ruta
            
            return rutas
        except Exception as e:
            self.logger.error(f"Error generando visualizaciones: {str(e)}")
            raise

    def generar_recomendaciones(self, analisis_datos, evaluacion, interpretacion):
        """Genera recomendaciones basadas en el análisis"""
        try:
            recomendaciones = []
            
            # 1. Recomendaciones basadas en datos
            if analisis_datos['valores_faltantes']['total'] > 0:
                recomendaciones.append({
                    'tipo': 'datos',
                    'mensaje': 'Considerar estrategias para manejar valores faltantes',
                    'detalles': analisis_datos['valores_faltantes']
                })
            
            # 2. Recomendaciones basadas en rendimiento
            if evaluacion['cross_validation']['std_cv']['test'] > 0.1:
                recomendaciones.append({
                    'tipo': 'rendimiento',
                    'mensaje': 'Alta variabilidad en validación cruzada, considerar más datos o regularización',
                    'detalles': evaluacion['cross_validation']
                })
            
            # 3. Recomendaciones basadas en interpretabilidad
            variables_importantes = interpretacion['global']['importancia_variables']
            if len(variables_importantes) > 10:
                recomendaciones.append({
                    'tipo': 'interpretabilidad',
                    'mensaje': 'Considerar selección de características para simplificar el modelo',
                    'detalles': {
                        'num_variables': len(variables_importantes),
                        'top_variables': variables_importantes.head().to_dict()
                    }
                })
            
            return recomendaciones
        except Exception as e:
            self.logger.error(f"Error generando recomendaciones: {str(e)}")
            raise

    def _calcular_estadisticas(self, datos):
        """Calcula estadísticas básicas del dataset"""
        return {
            'shape': datos.shape,
            'dtypes': datos.dtypes.value_counts().to_dict(),
            'memory_usage': datos.memory_usage().sum() / 1024**2,  # MB
            'estadisticas': datos.describe().to_dict()
        }

    def _analizar_distribucion(self, serie):
        """Analiza la distribución de una serie"""
        return {
            'tipo': 'categorica' if serie.dtype == 'object' else 'numerica',
            'unique_values': serie.nunique(),
            'stats': {
                'mean': float(serie.mean()) if serie.dtype != 'object' else None,
                'std': float(serie.std()) if serie.dtype != 'object' else None,
                'skew': float(serie.skew()) if serie.dtype != 'object' else None,
                'kurtosis': float(serie.kurtosis()) if serie.dtype != 'object' else None
            }
        }

    def _analizar_correlaciones(self, datos, objetivo):
        """Analiza correlaciones con la variable objetivo"""
        if datos[objetivo].dtype != 'object':
            correlaciones = datos.corr()[objetivo].sort_values(ascending=False)
            return correlaciones.to_dict()
        return {}

    def _analizar_valores_faltantes(self, datos):
        """Analiza valores faltantes en el dataset"""
        nulos = datos.isnull().sum()
        return {
            'total': int(nulos.sum()),
            'por_columna': nulos[nulos > 0].to_dict(),
            'porcentaje': float((nulos.sum() / (datos.shape[0] * datos.shape[1])) * 100)
        }

    def _identificar_casos_interesantes(self, modelo, datos, objetivo, n_casos=5):
        """Identifica casos interesantes para análisis local"""
        try:
            X = datos.drop(columns=[objetivo])
            y = datos[objetivo]
            y_pred = modelo.predict(X)
            
            if hasattr(modelo, 'predict_proba'):
                # Para clasificación, buscar casos con baja confianza
                y_proba = modelo.predict_proba(X)
                confianza = np.max(y_proba, axis=1)
                return np.argsort(confianza)[:n_casos]
            else:
                # Para regresión, buscar mayores errores
                errores = np.abs(y - y_pred)
                return np.argsort(errores)[-n_casos:]
                
        except Exception as e:
            self.logger.error(f"Error identificando casos interesantes: {str(e)}")
            raise 