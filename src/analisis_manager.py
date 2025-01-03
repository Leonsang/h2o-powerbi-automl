import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class AnalisisManager:
    def __init__(self, usar_seaborn=True):
        self.usar_seaborn = usar_seaborn
        if usar_seaborn:
            try:
                import seaborn as sns
                sns.set_style("whitegrid")
            except ImportError:
                self.usar_seaborn = False
        
        self.tipos_modelo = [
            'automl', 'gbm', 'rf', 'glm', 'deeplearning', 
            'xgboost', 'lightgbm', 'stackedensemble', 'drf'
        ]
        
    def generar_reporte_completo(self, modelo, datos, predicciones):
        """Genera reporte con o sin visualizaciones según disponibilidad de seaborn"""
        reporte = {}
        
        # Métricas básicas (sin seaborn)
        reporte['metricas'] = self._calcular_metricas_basicas(modelo, datos, predicciones)
        
        # Visualizaciones (solo si seaborn está disponible)
        if self.usar_seaborn:
            reporte['graficos'] = self._generar_visualizaciones(modelo, datos, predicciones)
            
        return reporte 