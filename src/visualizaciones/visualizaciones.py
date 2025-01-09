import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os

logger = logging.getLogger(__name__)

class Visualizador:
    """Clase para generar visualizaciones avanzadas"""
    
    def __init__(self, estilo='plotly'):
        self.estilo = estilo
        self.paleta_colores = px.colors.qualitative.Set3
        self.configuracion = {
            'template': 'plotly_white',
            'height': 600,
            'width': 800
        }
        # Asegurar que el directorio existe
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output', 'visualizaciones'))
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Directorio de visualizaciones: {self.output_dir}")
    
    def plot_distribucion_variables(self, df: pd.DataFrame, columnas: list = None) -> go.Figure:
        """Genera gráficos de distribución para variables"""
        try:
            if columnas is None:
                columnas = df.columns
            
            # Calcular número de filas y columnas para subplots
            n_vars = len(columnas)
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            
            # Crear figura con subplots
            fig = make_subplots(rows=n_rows, cols=n_cols, 
                              subplot_titles=[f'Distribución de {col}' for col in columnas])
            
            # Agregar cada variable como un subplot
            for i, col in enumerate(columnas):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                if df[col].dtype in ['int64', 'float64']:
                    # Distribución numérica
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col),
                        row=row, col=col_pos
                    )
                else:
                    # Distribución categórica
                    conteos = df[col].value_counts()
                    fig.add_trace(
                        go.Bar(x=conteos.index, y=conteos.values, name=col),
                        row=row, col=col_pos
                    )
            
            # Actualizar layout
            fig.update_layout(
                title='Distribución de Variables',
                showlegend=False,
                height=300 * n_rows,
                width=300 * n_cols,
                template='plotly_white'
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error generando distribuciones: {str(e)}")
            return None
    
    def plot_correlaciones(self, df: pd.DataFrame) -> go.Figure:
        """Genera matriz de correlaciones"""
        try:
            # Seleccionar solo variables numéricas
            df_num = df.select_dtypes(include=['int64', 'float64'])
            
            # Calcular correlaciones
            corr = df_num.corr()
            
            # Crear heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                title='Matriz de Correlaciones',
                **self.configuracion
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error generando correlaciones: {str(e)}")
            return None
    
    def plot_importancia_variables(self, importancia: dict) -> go.Figure:
        """Visualiza importancia de variables"""
        try:
            # Convertir el diccionario a DataFrame de manera explícita
            if isinstance(importancia, dict):
                vars_imp = pd.DataFrame([
                    {'variable': k, 'importance': v} 
                    for k, v in importancia.items()
                ])
            else:
                vars_imp = pd.DataFrame(importancia)
            
            # Ordenar variables por importancia
            vars_imp = vars_imp.sort_values('importance', ascending=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=vars_imp['importance'],
                    y=vars_imp['variable'],
                    orientation='h'
                )
            ])
            
            fig.update_layout(
                title='Importancia de Variables',
                xaxis_title='Importancia',
                yaxis_title='Variable',
                **self.configuracion
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error visualizando importancia: {str(e)}")
            return None
    
    def plot_metricas_tiempo(self, metricas: pd.DataFrame) -> go.Figure:
        """Visualiza evolución de métricas en el tiempo"""
        try:
            fig = go.Figure()
            
            for col in metricas.columns:
                if col != 'fecha':
                    fig.add_trace(go.Scatter(
                        x=metricas['fecha'],
                        y=metricas[col],
                        name=col,
                        mode='lines+markers'
                    ))
            
            fig.update_layout(
                title='Evolución de Métricas',
                xaxis_title='Fecha',
                yaxis_title='Valor',
                **self.configuracion
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error visualizando métricas: {str(e)}")
            return None
    
    def plot_predicciones_vs_real(self, y_true: np.array, y_pred: np.array) -> go.Figure:
        """Visualiza predicciones vs valores reales"""
        try:
            fig = go.Figure()
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predicciones'
            ))
            
            # Línea ideal
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Ideal',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title='Predicciones vs Valores Reales',
                xaxis_title='Valores Reales',
                yaxis_title='Predicciones',
                **self.configuracion
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error visualizando predicciones: {str(e)}")
            return None
    
    def plot_residuos(self, y_true: np.array, y_pred: np.array) -> go.Figure:
        """Visualiza análisis de residuos"""
        try:
            residuos = y_true - y_pred
            
            # Crear subplots
            fig = make_subplots(rows=2, cols=1)
            
            # Histograma de residuos
            fig.add_trace(
                go.Histogram(x=residuos, name='Residuos'),
                row=1, col=1
            )
            
            # Scatter plot de residuos vs predicciones
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuos,
                    mode='markers',
                    name='Residuos vs Predicciones'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Análisis de Residuos',
                **self.configuracion
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error visualizando residuos: {str(e)}")
            return None
    
    def guardar_visualizacion(self, fig: go.Figure, nombre: str, modelo_id: str = None) -> str:
        """Guarda una visualización en el directorio de salida"""
        try:
            # Si se proporciona modelo_id, crear subdirectorio para el modelo
            if modelo_id:
                output_dir = os.path.join(self.output_dir, modelo_id)
            else:
                output_dir = self.output_dir
                
            # Asegurar que el directorio existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Construir ruta del archivo
            ruta_archivo = os.path.join(output_dir, f"{nombre}.html")
            
            # Guardar visualización
            fig.write_html(ruta_archivo)
            logger.info(f"Visualización guardada en: {ruta_archivo}")
            
            return ruta_archivo
        except Exception as e:
            logger.error(f"Error guardando visualización {nombre}: {str(e)}")
            return None
    
    def generar_reporte_visual(self, df: pd.DataFrame, resultados: dict) -> dict:
        """Genera un reporte visual completo"""
        try:
            reporte = {}
            
            # Generar visualizaciones
            visualizaciones = {
                'distribucion': self.plot_distribucion_variables(df),
                'correlaciones': self.plot_correlaciones(df),
                'importancia': self.plot_importancia_variables(resultados.get('importancia_variables', {})),
                'metricas': self.plot_metricas_tiempo(resultados.get('metricas_tiempo', pd.DataFrame())),
                'predicciones': self.plot_predicciones_vs_real(
                    resultados.get('y_true', np.array([])),
                    resultados.get('y_pred', np.array([]))
                ),
                'residuos': self.plot_residuos(
                    resultados.get('y_true', np.array([])),
                    resultados.get('y_pred', np.array([]))
                )
            }
            
            # Guardar visualizaciones
            for nombre, fig in visualizaciones.items():
                if fig is not None:
                    ruta = self.guardar_visualizacion(fig, nombre)
                    reporte[nombre] = ruta
            
            return reporte
        except Exception as e:
            logger.error(f"Error generando reporte visual: {str(e)}")
            return {} 