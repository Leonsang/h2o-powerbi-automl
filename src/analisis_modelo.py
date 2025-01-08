import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import dice_ml
from src.logger import Logger
from src.asistente_ia import AsistenteDataScience

logger = Logger('analisis_modelo')

class AnalizadorModelo:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.logger = logger
        self.asistente = AsistenteDataScience()

    def analisis_completo(self, modelo, datos, predicciones, objetivo, tipo_modelo):
        """Genera análisis completo del modelo"""
        try:
            self.logger.info("Iniciando análisis completo del modelo")
            
            # 1. Análisis Global
            analisis_global = self.generar_analisis_global(
                modelo, datos, predicciones, objetivo
            )
            
            # 2. Análisis Local
            analisis_local = self.generar_analisis_local(
                modelo, datos, predicciones
            )
            
            # 3. Visualizaciones
            graficos = self.generar_visualizaciones(
                datos, predicciones, objetivo,
                analisis_global, analisis_local
            )
            
            # 4. Insights
            insights = self.generar_insights(
                datos, predicciones, 
                analisis_global, analisis_local
            )
            
            # Añadir análisis IA
            analisis_ia = self.asistente.interpretar_resultados({
                'metricas': analisis_global['metricas'],
                'importancia_variables': analisis_global['importancia_variables'],
                'insights': insights
            })
            
            recomendaciones_tecnicas = self.asistente.generar_recomendaciones_tecnicas({
                'metricas': analisis_global['metricas'],
                'shap_values': analisis_global['shap_values']
            })
            
            # Añadir explicaciones IA a casos específicos
            for caso in analisis_local['lime_explicaciones']:
                caso['explicacion_ia'] = self.asistente.explicar_predicciones(
                    caso['caso'],
                    caso['explicacion'],
                    analisis_local['counterfactuals'][0]  # Tomar el primer counterfactual
                )
            
            self.logger.info("Análisis completo generado exitosamente")
            return {
                'analisis_global': analisis_global,
                'analisis_local': analisis_local,
                'graficos': graficos,
                'insights': insights,
                'analisis_ia': {
                    'interpretacion_general': analisis_ia,
                    'recomendaciones_tecnicas': recomendaciones_tecnicas
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis completo: {str(e)}")
            raise

    def generar_analisis_global(self, modelo, datos, predicciones, objetivo):
        """Genera análisis global del modelo"""
        try:
            # 1. Importancia de variables
            importancia = self.calcular_importancia_variables(modelo, datos)
            
            # 2. SHAP values
            shap_values = self.calcular_shap_values(modelo, datos)
            
            # 3. Dependencias parciales
            pdp_plots = self.generar_pdp_plots(modelo, datos)
            
            # 4. Métricas globales
            metricas = self.calcular_metricas_globales(
                datos[objetivo], predicciones
            )
            
            return {
                'importancia_variables': importancia,
                'shap_values': shap_values,
                'pdp_plots': pdp_plots,
                'metricas': metricas
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis global: {str(e)}")
            raise

    def generar_analisis_local(self, modelo, datos, predicciones):
        """Genera análisis local para explicaciones individuales"""
        try:
            # 1. LIME explicaciones
            lime_exp = self.generar_lime_explicaciones(modelo, datos)
            
            # 2. Counterfactuals
            counterfactuals = self.generar_counterfactuals(modelo, datos)
            
            # 3. Análisis de casos específicos
            casos_especiales = self.analizar_casos_especiales(
                datos, predicciones
            )
            
            return {
                'lime_explicaciones': lime_exp,
                'counterfactuals': counterfactuals,
                'casos_especiales': casos_especiales
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis local: {str(e)}")
            raise

    def generar_visualizaciones(self, datos, predicciones, objetivo, 
                              analisis_global, analisis_local):
        """Genera todas las visualizaciones necesarias"""
        try:
            graficos = {}
            
            # 1. Predicciones vs Reales
            fig = self.plot_predicciones_vs_reales(
                datos[objetivo], predicciones
            )
            fig.savefig(f"{self.output_dir}/predicciones_vs_reales.png")
            graficos['predicciones_vs_reales'] = fig
            
            # 2. SHAP Summary
            fig = self.plot_shap_summary(
                analisis_global['shap_values']
            )
            fig.savefig(f"{self.output_dir}/shap_summary.png")
            graficos['shap_summary'] = fig
            
            # 3. Importancia de Variables
            fig = self.plot_importancia_variables(
                analisis_global['importancia_variables']
            )
            fig.savefig(f"{self.output_dir}/importancia_variables.png")
            graficos['importancia_variables'] = fig
            
            # 4. Distribución de Errores
            fig = self.plot_distribucion_errores(
                datos[objetivo], predicciones
            )
            fig.savefig(f"{self.output_dir}/distribucion_errores.png")
            graficos['distribucion_errores'] = fig
            
            return graficos
            
        except Exception as e:
            self.logger.error(f"Error generando visualizaciones: {str(e)}")
            raise

    def generar_insights(self, datos, predicciones, analisis_global, analisis_local):
        """Genera insights automáticos del modelo"""
        try:
            insights = []
            
            # 1. Variables más importantes
            top_vars = self.obtener_top_variables(
                analisis_global['importancia_variables']
            )
            insights.append({
                'tipo': 'importancia',
                'mensaje': f"Las variables más influyentes son: {top_vars}"
            })
            
            # 2. Patrones detectados
            patrones = self.detectar_patrones(
                analisis_global['shap_values'], datos
            )
            insights.append({
                'tipo': 'patrones',
                'mensaje': f"Patrones principales detectados: {patrones}"
            })
            
            # 3. Anomalías
            anomalias = self.detectar_anomalias(datos, predicciones)
            insights.append({
                'tipo': 'anomalias',
                'mensaje': f"Se detectaron {len(anomalias)} casos anómalos"
            })
            
            # 4. Recomendaciones
            recomendaciones = self.generar_recomendaciones(
                analisis_global['shap_values'], datos
            )
            insights.append({
                'tipo': 'recomendaciones',
                'mensaje': f"Recomendaciones principales: {recomendaciones}"
            })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generando insights: {str(e)}")
            raise 