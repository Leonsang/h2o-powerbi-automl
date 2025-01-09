import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.logger import Logger
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = Logger('visualizaciones')

class VisualizacionesManager:
    """Gestiona todas las visualizaciones del modelo y análisis"""
    
    def __init__(self, output_dir='./output/graficos'):
        self.logger = logger
        self.output_dir = output_dir
        self.config_plots()

    def config_plots(self):
        """Configura el estilo de los plots"""
        plt.style.use('seaborn')
        sns.set_theme(style="whitegrid")
        
    def visualizar_distribucion_objetivo(self, datos, objetivo, tipo_modelo='regresion'):
        """Visualiza la distribución de la variable objetivo"""
        try:
            fig = plt.figure(figsize=(10, 6))
            
            if tipo_modelo == 'regresion':
                sns.histplot(data=datos, x=objetivo, kde=True)
                plt.title(f'Distribución de {objetivo}')
            else:
                sns.countplot(data=datos, x=objetivo)
                plt.title(f'Distribución de clases en {objetivo}')
                
            plt.xlabel(objetivo)
            plt.ylabel('Frecuencia')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error en visualización de distribución: {str(e)}")
            raise

    def visualizar_correlaciones(self, datos, objetivo=None):
        """Genera matriz de correlaciones"""
        try:
            # Seleccionar solo columnas numéricas
            numericas = datos.select_dtypes(include=[np.number])
            
            # Calcular correlaciones
            corr = numericas.corr()
            
            # Crear máscara para triángulo superior
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Generar heatmap
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, ax=ax)
            
            plt.title('Matriz de Correlaciones')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error en visualización de correlaciones: {str(e)}")
            raise

    def visualizar_importancia_variables(self, importancia, top_n=10):
        """Visualiza la importancia de las variables"""
        try:
            # Tomar top N variables
            top_vars = importancia.head(top_n)
            
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(data=top_vars, x='importance', y='feature')
            
            plt.title(f'Top {top_n} Variables más Importantes')
            plt.xlabel('Importancia')
            plt.ylabel('Variable')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error en visualización de importancia: {str(e)}")
            raise

    def visualizar_predicciones_vs_real(self, y_true, y_pred, tipo_modelo='regresion'):
        """Visualiza predicciones vs valores reales"""
        try:
            fig = plt.figure(figsize=(10, 6))
            
            if tipo_modelo == 'regresion':
                plt.scatter(y_true, y_pred, alpha=0.5)
                plt.plot([y_true.min(), y_true.max()], 
                        [y_true.min(), y_true.max()], 
                        'r--', lw=2)
                
                plt.xlabel('Valores Reales')
                plt.ylabel('Predicciones')
                plt.title('Predicciones vs Valores Reales')
            else:
                confusion = pd.crosstab(y_true, y_pred)
                sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicciones')
                plt.ylabel('Valores Reales')
                plt.title('Matriz de Confusión')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error en visualización de predicciones: {str(e)}")
            raise

    def visualizar_residuos(self, residuos):
        """Visualiza análisis de residuos"""
        try:
            fig = make_subplots(rows=2, cols=2,
                              subplot_titles=('Distribución de Residuos',
                                            'Q-Q Plot',
                                            'Residuos vs Predicciones',
                                            'Residuos Estandarizados'))
            
            # Distribución
            fig.add_trace(
                go.Histogram(x=residuos, name='Residuos'),
                row=1, col=1
            )
            
            # Q-Q Plot
            qq = np.percentile(residuos, [i for i in range(0, 101)])
            theoretical_q = np.percentile(np.random.normal(0, 1, len(residuos)), 
                                        [i for i in range(0, 101)])
            
            fig.add_trace(
                go.Scatter(x=theoretical_q, y=qq, mode='markers',
                          name='Q-Q Plot'),
                row=1, col=2
            )
            
            # Residuos vs Predicciones
            fig.add_trace(
                go.Scatter(x=range(len(residuos)), y=residuos,
                          mode='markers', name='Residuos vs Index'),
                row=2, col=1
            )
            
            # Residuos Estandarizados
            std_residuos = (residuos - np.mean(residuos)) / np.std(residuos)
            fig.add_trace(
                go.Scatter(x=range(len(std_residuos)), y=std_residuos,
                          mode='markers', name='Residuos Estandarizados'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False,
                            title_text="Análisis de Residuos")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error en visualización de residuos: {str(e)}")
            raise

    def visualizar_shap_summary(self, shap_values, feature_names):
        """Visualiza resumen de valores SHAP"""
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            fig = plt.gcf()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error en visualización SHAP: {str(e)}")
            raise

    def visualizar_pdp(self, pdp_results, feature):
        """Visualiza gráfico de dependencia parcial"""
        try:
            fig = plt.figure(figsize=(8, 6))
            
            plt.plot(pdp_results[feature]['feature_values'],
                    pdp_results[feature]['values'])
            
            plt.xlabel(feature)
            plt.ylabel('Impacto Parcial')
            plt.title(f'Gráfico de Dependencia Parcial - {feature}')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error en visualización PDP: {str(e)}")
            raise

    def guardar_visualizacion(self, fig, nombre):
        """Guarda la visualización en el directorio de salida"""
        try:
            ruta = f"{self.output_dir}/{nombre}.png"
            fig.savefig(ruta, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return ruta
        except Exception as e:
            self.logger.error(f"Error guardando visualización: {str(e)}")
            raise 