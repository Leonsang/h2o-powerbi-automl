import pandas as pd
import numpy as np
from src.logger import Logger

logger = Logger('asistente_ia')

class AsistenteIA:
    """
    Asistente de IA para interpretación avanzada de modelos y resultados
    """
    
    def __init__(self):
        self.logger = logger
        self.contexto_modelo = {}

    def interpretar_modelo(self, modelo_info, datos_analisis):
        """
        Interpreta los resultados del modelo de manera inteligente
        
        Args:
            modelo_info: Información del modelo y sus capacidades
            datos_analisis: Resultados del análisis (métricas, shap, etc.)
        """
        try:
            interpretacion = {
                'resumen_ejecutivo': self._generar_resumen_ejecutivo(modelo_info, datos_analisis),
                'insights_clave': self._identificar_insights(datos_analisis),
                'recomendaciones': self._generar_recomendaciones(datos_analisis),
                'explicaciones_detalladas': self._generar_explicaciones_detalladas(modelo_info, datos_analisis)
            }
            return interpretacion
        except Exception as e:
            self.logger.error(f"Error en interpretación de modelo: {str(e)}")
            raise

    def interpretar_prediccion(self, prediccion, shap_values, feature_names):
        """
        Genera una explicación en lenguaje natural de una predicción específica
        """
        try:
            # Identificar factores más importantes
            importancia = np.abs(shap_values)
            top_indices = np.argsort(importancia)[-5:]  # Top 5 factores
            
            # Generar explicación
            explicacion = "Esta predicción se basa principalmente en:\n"
            
            for idx in top_indices:
                feature = feature_names[idx]
                impacto = shap_values[idx]
                direccion = "aumenta" if impacto > 0 else "disminuye"
                
                explicacion += f"- {feature}: {direccion} la predicción en {abs(impacto):.2f} unidades\n"
            
            return {
                'explicacion_natural': explicacion,
                'factores_clave': [feature_names[i] for i in top_indices],
                'impactos': shap_values[top_indices].tolist()
            }
        except Exception as e:
            self.logger.error(f"Error en interpretación de predicción: {str(e)}")
            raise

    def analizar_comportamiento_modelo(self, modelo_info, metricas_historicas):
        """
        Analiza el comportamiento del modelo a lo largo del tiempo
        """
        try:
            analisis = {
                'tendencias': self._analizar_tendencias(metricas_historicas),
                'alertas': self._detectar_alertas(metricas_historicas),
                'recomendaciones_mejora': self._recomendar_mejoras(modelo_info, metricas_historicas)
            }
            return analisis
        except Exception as e:
            self.logger.error(f"Error en análisis de comportamiento: {str(e)}")
            raise

    def generar_narrativa_analisis(self, resultados_analisis):
        """
        Genera una narrativa coherente del análisis completo
        """
        try:
            narrativa = {
                'resumen': self._generar_resumen_narrativo(resultados_analisis),
                'hallazgos_principales': self._identificar_hallazgos(resultados_analisis),
                'conclusiones': self._generar_conclusiones(resultados_analisis),
                'siguientes_pasos': self._recomendar_siguientes_pasos(resultados_analisis)
            }
            return narrativa
        except Exception as e:
            self.logger.error(f"Error generando narrativa: {str(e)}")
            raise

    def _generar_resumen_ejecutivo(self, modelo_info, datos_analisis):
        """Genera un resumen ejecutivo del modelo y sus resultados"""
        try:
            # Extraer información relevante
            tipo_modelo = modelo_info.get('tipo', 'No especificado')
            metricas = datos_analisis.get('metricas', {})
            
            # Generar resumen
            resumen = {
                'tipo_modelo': tipo_modelo,
                'rendimiento_general': self._evaluar_rendimiento(metricas),
                'fortalezas': self._identificar_fortalezas(datos_analisis),
                'areas_mejora': self._identificar_areas_mejora(datos_analisis),
                'confiabilidad': self._evaluar_confiabilidad(datos_analisis)
            }
            
            return resumen
        except Exception as e:
            self.logger.error(f"Error generando resumen ejecutivo: {str(e)}")
            raise

    def _identificar_insights(self, datos_analisis):
        """Identifica insights clave del análisis"""
        try:
            insights = []
            
            # Análisis de importancia de variables
            if 'importancia_variables' in datos_analisis:
                top_vars = datos_analisis['importancia_variables'].head(3)
                insights.append({
                    'tipo': 'variables_importantes',
                    'descripcion': f"Las variables más influyentes son: {', '.join(top_vars['feature'])}",
                    'impacto': 'alto'
                })
            
            # Análisis de patrones
            if 'shap_values' in datos_analisis:
                patrones = self._detectar_patrones(datos_analisis['shap_values'])
                insights.extend(patrones)
            
            # Análisis de comportamiento
            if 'predicciones' in datos_analisis:
                comportamiento = self._analizar_comportamiento_predicciones(datos_analisis['predicciones'])
                insights.extend(comportamiento)
            
            return insights
        except Exception as e:
            self.logger.error(f"Error identificando insights: {str(e)}")
            raise

    def _generar_recomendaciones(self, datos_analisis):
        """Genera recomendaciones basadas en el análisis"""
        try:
            recomendaciones = []
            
            # Recomendaciones de variables
            if 'importancia_variables' in datos_analisis:
                vars_bajas = datos_analisis['importancia_variables'].tail(5)
                if len(vars_bajas) > 0:
                    recomendaciones.append({
                        'tipo': 'optimizacion_variables',
                        'descripcion': 'Considerar eliminar variables de baja importancia',
                        'detalles': vars_bajas['feature'].tolist()
                    })
            
            # Recomendaciones de rendimiento
            if 'metricas' in datos_analisis:
                rec_rendimiento = self._generar_recomendaciones_rendimiento(datos_analisis['metricas'])
                recomendaciones.extend(rec_rendimiento)
            
            # Recomendaciones de datos
            if 'analisis_datos' in datos_analisis:
                rec_datos = self._generar_recomendaciones_datos(datos_analisis['analisis_datos'])
                recomendaciones.extend(rec_datos)
            
            return recomendaciones
        except Exception as e:
            self.logger.error(f"Error generando recomendaciones: {str(e)}")
            raise

    def _generar_explicaciones_detalladas(self, modelo_info, datos_analisis):
        """Genera explicaciones detalladas del modelo y sus predicciones"""
        try:
            explicaciones = {
                'modelo': self._explicar_funcionamiento_modelo(modelo_info),
                'variables': self._explicar_variables(datos_analisis),
                'predicciones': self._explicar_predicciones(datos_analisis),
                'limitaciones': self._identificar_limitaciones(modelo_info, datos_analisis)
            }
            return explicaciones
        except Exception as e:
            self.logger.error(f"Error generando explicaciones detalladas: {str(e)}")
            raise

    def _evaluar_rendimiento(self, metricas):
        """Evalúa el rendimiento general del modelo"""
        try:
            evaluacion = {
                'nivel': 'No determinado',
                'score': 0.0,
                'detalles': []
            }
            
            if 'r2' in metricas:
                score = metricas['r2']
                if score > 0.9:
                    evaluacion.update({'nivel': 'Excelente', 'score': score})
                elif score > 0.7:
                    evaluacion.update({'nivel': 'Bueno', 'score': score})
                else:
                    evaluacion.update({'nivel': 'Necesita mejoras', 'score': score})
            
            elif 'accuracy' in metricas:
                score = metricas['accuracy']
                if score > 0.9:
                    evaluacion.update({'nivel': 'Excelente', 'score': score})
                elif score > 0.7:
                    evaluacion.update({'nivel': 'Bueno', 'score': score})
                else:
                    evaluacion.update({'nivel': 'Necesita mejoras', 'score': score})
            
            return evaluacion
        except Exception as e:
            self.logger.error(f"Error evaluando rendimiento: {str(e)}")
            raise

    def _detectar_patrones(self, shap_values):
        """Detecta patrones en los valores SHAP"""
        try:
            patrones = []
            
            # Analizar interacciones entre variables
            if isinstance(shap_values, dict) and 'values' in shap_values:
                valores = shap_values['values']
                
                # Detectar variables con alto impacto consistente
                impactos_medios = np.mean(np.abs(valores), axis=0)
                vars_importantes = np.where(impactos_medios > np.percentile(impactos_medios, 90))[0]
                
                for var_idx in vars_importantes:
                    patron = {
                        'tipo': 'impacto_consistente',
                        'variable_idx': int(var_idx),
                        'impacto_medio': float(impactos_medios[var_idx])
                    }
                    patrones.append(patron)
            
            return patrones
        except Exception as e:
            self.logger.error(f"Error detectando patrones: {str(e)}")
            raise 