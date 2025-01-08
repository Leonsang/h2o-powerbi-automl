import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.logger import Logger

logger = Logger('visualizaciones')

class Visualizador:
    def plot_predicciones_vs_reales(self, reales, predicciones):
        """Gráfico de dispersión de predicciones vs valores reales"""
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(reales, predicciones, alpha=0.5)
            plt.plot([reales.min(), reales.max()], 
                    [reales.min(), reales.max()], 'r--')
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title('Predicciones vs Valores Reales')
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error en plot predicciones vs reales: {str(e)}")
            raise

    def plot_shap_summary(self, shap_values):
        """Gráfico resumen de valores SHAP"""
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, plot_type="bar")
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error en plot SHAP summary: {str(e)}")
            raise

    def plot_importancia_variables(self, importancia):
        """Gráfico de barras de importancia de variables"""
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importancia)
            plt.xticks(rotation=45)
            plt.title('Importancia de Variables')
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error en plot importancia variables: {str(e)}")
            raise

    def plot_distribucion_errores(self, reales, predicciones):
        """Histograma de distribución de errores"""
        try:
            errores = reales - predicciones
            plt.figure(figsize=(10, 6))
            sns.histplot(errores, kde=True)
            plt.title('Distribución de Errores')
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error en plot distribución errores: {str(e)}")
            raise 