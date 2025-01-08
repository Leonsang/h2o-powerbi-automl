import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class AnalisisManager:
    def __init__(self, ruta_base="src/modelos"):
        self.ruta_base = ruta_base
        
    def guardar_analisis(self, analisis, modelo_id, timestamp=None):
        """Guarda análisis completo del modelo"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
        ruta_analisis = os.path.join(
            self.ruta_base, 
            timestamp, 
            modelo_id, 
            'analisis'
        )
        os.makedirs(ruta_analisis, exist_ok=True)
        
        # 1. Guardar métricas
        self._guardar_metricas(analisis['metricas'], ruta_analisis)
        
        # 2. Guardar gráficos
        self._generar_graficos(analisis, ruta_analisis)
        
        # 3. Guardar análisis detallado
        self._guardar_analisis_detallado(analisis, ruta_analisis)
        
    def _guardar_metricas(self, metricas, ruta):
        """Guarda métricas en formato JSON"""
        with open(os.path.join(ruta, 'metricas.json'), 'w') as f:
            json.dump(metricas, f, indent=4)
            
    def _generar_graficos(self, analisis, ruta):
        """Genera y guarda todos los gráficos"""
        ruta_graficos = os.path.join(ruta, 'graficos')
        os.makedirs(ruta_graficos, exist_ok=True)
        
        # 1. Importancia de variables
        if 'importancia_variables' in analisis:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=analisis['importancia_variables'])
            plt.xticks(rotation=45)
            plt.title('Importancia de Variables')
            plt.tight_layout()
            plt.savefig(os.path.join(ruta_graficos, 'importancia_variables.png'))
            plt.close()
            
        # 2. Distribución de errores
        if 'analisis_errores' in analisis:
            plt.figure(figsize=(10, 6))
            sns.histplot(analisis['analisis_errores']['distribucion'])
            plt.title('Distribución de Errores')
            plt.savefig(os.path.join(ruta_graficos, 'distribucion_errores.png'))
            plt.close()
            
        # 3. Tendencias
        if 'tendencias' in analisis:
            self._graficar_tendencias(
                analisis['tendencias'], 
                ruta_graficos
            )
            
    def _graficar_tendencias(self, tendencias, ruta):
        """Genera gráficos de tendencias"""
        if 'estacionalidad' in tendencias:
            componentes = tendencias['estacionalidad']
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            
            axes[0].plot(componentes['trend'])
            axes[0].set_title('Tendencia')
            
            axes[1].plot(componentes['seasonal'])
            axes[1].set_title('Estacionalidad')
            
            axes[2].plot(componentes['resid'])
            axes[2].set_title('Residuos')
            
            plt.tight_layout()
            plt.savefig(os.path.join(ruta, 'descomposicion_temporal.png'))
            plt.close()
            
    def _guardar_analisis_detallado(self, analisis, ruta):
        """Guarda análisis detallado en formato CSV"""
        pd.DataFrame(analisis).to_csv(
            os.path.join(ruta, 'analisis_detallado.csv')
        ) 