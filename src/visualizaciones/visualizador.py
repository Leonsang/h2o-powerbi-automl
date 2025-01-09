import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Union
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Visualizador:
    """Clase para generar visualizaciones de datos y modelos"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el visualizador
        
        Args:
            config: Configuración del visualizador
        """
        self.config = config or {}
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output', 'visualizaciones'))
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_distribucion_variables(self, df: pd.DataFrame, vars_numericas: Optional[List[str]] = None) -> go.Figure:
        """
        Genera gráficos de distribución para variables numéricas
        
        Args:
            df: DataFrame con los datos
            vars_numericas: Lista de variables numéricas a visualizar
            
        Returns:
            Figura con las distribuciones
        """
        try:
            # Si no se especifican variables, usar todas las numéricas
            if vars_numericas is None:
                vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calcular número de filas y columnas para subplots
            n_vars = len(vars_numericas)
            n_cols = min(3, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            
            # Crear figura con subplots
            fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=vars_numericas)
            
            # Agregar histogramas
            for i, var in enumerate(vars_numericas):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=df[var], name=var),
                    row=row, col=col
                )
            
            # Actualizar layout
            fig.update_layout(
                height=300*n_rows,
                width=300*n_cols,
                showlegend=False,
                title_text="Distribución de Variables Numéricas"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generando distribuciones: {str(e)}")
            raise
    
    def plot_correlacion(self, df: pd.DataFrame, vars_numericas: Optional[List[str]] = None) -> go.Figure:
        """
        Genera matriz de correlación
        
        Args:
            df: DataFrame con los datos
            vars_numericas: Lista de variables numéricas a visualizar
            
        Returns:
            Figura con la matriz de correlación
        """
        try:
            # Si no se especifican variables, usar todas las numéricas
            if vars_numericas is None:
                vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calcular correlaciones
            corr = df[vars_numericas].corr()
            
            # Crear heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmin=-1, zmax=1
            ))
            
            # Actualizar layout
            fig.update_layout(
                title_text="Matriz de Correlación",
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generando correlaciones: {str(e)}")
            raise
    
    def plot_importancia_variables(self, importancias: Dict[str, float]) -> go.Figure:
        """
        Genera gráfico de importancia de variables
        
        Args:
            importancias: Diccionario con importancias de variables
            
        Returns:
            Figura con importancia de variables
        """
        try:
            # Ordenar importancias
            importancias_sorted = dict(sorted(importancias.items(), key=lambda x: x[1], reverse=True))
            
            # Crear gráfico de barras
            fig = go.Figure(data=go.Bar(
                x=list(importancias_sorted.keys()),
                y=list(importancias_sorted.values())
            ))
            
            # Actualizar layout
            fig.update_layout(
                title_text="Importancia de Variables",
                xaxis_title="Variables",
                yaxis_title="Importancia",
                height=400,
                width=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generando importancia de variables: {str(e)}")
            raise
    
    def plot_metricas_modelo(self, metricas: Dict[str, float]) -> go.Figure:
        """
        Genera gráfico de métricas del modelo
        
        Args:
            metricas: Diccionario con métricas del modelo
            
        Returns:
            Figura con métricas
        """
        try:
            # Filtrar métricas numéricas
            metricas_num = {k: v for k, v in metricas.items() 
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
            
            # Crear gráfico de barras
            fig = go.Figure(data=go.Bar(
                x=list(metricas_num.keys()),
                y=list(metricas_num.values())
            ))
            
            # Actualizar layout
            fig.update_layout(
                title_text="Métricas del Modelo",
                xaxis_title="Métrica",
                yaxis_title="Valor",
                height=400,
                width=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generando métricas del modelo: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, confusion_matrix: Dict[str, Any]) -> go.Figure:
        """
        Genera matriz de confusión
        
        Args:
            confusion_matrix: Diccionario con matriz de confusión
            
        Returns:
            Figura con matriz de confusión
        """
        try:
            # Convertir diccionario a DataFrame
            df = pd.DataFrame(confusion_matrix)
            
            # Crear heatmap
            fig = go.Figure(data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale='Blues'
            ))
            
            # Actualizar layout
            fig.update_layout(
                title_text="Matriz de Confusión",
                xaxis_title="Predicho",
                yaxis_title="Real",
                height=500,
                width=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generando matriz de confusión: {str(e)}")
            raise
    
    def guardar_visualizacion(self, fig: go.Figure, nombre: str, modelo_id: Optional[str] = None) -> str:
        """
        Guarda una visualización en HTML
        
        Args:
            fig: Figura a guardar
            nombre: Nombre del archivo
            modelo_id: ID del modelo (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        try:
            # Crear subdirectorio para modelo si se especifica
            if modelo_id:
                output_dir = os.path.join(self.output_dir, modelo_id)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = self.output_dir
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{nombre}_{timestamp}.html"
            filepath = os.path.join(output_dir, filename)
            
            # Guardar figura
            fig.write_html(filepath)
            
            logger.info(f"Visualización guardada en: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error guardando visualización: {str(e)}")
            raise
    
    def generar_reporte_visual(self, df: pd.DataFrame, metricas: Dict[str, Any], 
                             importancias: Optional[Dict[str, float]] = None,
                             modelo_id: Optional[str] = None) -> Dict[str, str]:
        """
        Genera un reporte visual completo
        
        Args:
            df: DataFrame con los datos
            metricas: Métricas del modelo
            importancias: Importancia de variables (opcional)
            modelo_id: ID del modelo (opcional)
            
        Returns:
            Diccionario con rutas de las visualizaciones
        """
        try:
            visualizaciones = {}
            
            # Distribución de variables
            fig_dist = self.plot_distribucion_variables(df)
            visualizaciones['distribucion'] = self.guardar_visualizacion(
                fig_dist, 'distribucion', modelo_id
            )
            
            # Correlaciones
            fig_corr = self.plot_correlacion(df)
            visualizaciones['correlacion'] = self.guardar_visualizacion(
                fig_corr, 'correlacion', modelo_id
            )
            
            # Métricas del modelo
            fig_metricas = self.plot_metricas_modelo(metricas)
            visualizaciones['metricas'] = self.guardar_visualizacion(
                fig_metricas, 'metricas', modelo_id
            )
            
            # Matriz de confusión si existe
            if 'confusion_matrix' in metricas:
                fig_conf = self.plot_confusion_matrix(metricas['confusion_matrix'])
                visualizaciones['confusion_matrix'] = self.guardar_visualizacion(
                    fig_conf, 'confusion_matrix', modelo_id
                )
            
            # Importancia de variables si se proporciona
            if importancias:
                fig_imp = self.plot_importancia_variables(importancias)
                visualizaciones['importancia'] = self.guardar_visualizacion(
                    fig_imp, 'importancia', modelo_id
                )
            
            return visualizaciones
            
        except Exception as e:
            logger.error(f"Error generando reporte visual: {str(e)}")
            raise 