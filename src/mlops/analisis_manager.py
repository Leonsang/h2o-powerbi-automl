import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.logger import Logger

logger = Logger('analisis_manager')

class AnalisisManager:
    def __init__(self, ruta_base="src/modelos"):
        self.logger = logger
        self.ruta_base = ruta_base
        
    def guardar_analisis(self, analisis, modelo_id, timestamp=None):
        """Guarda análisis completo del modelo"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                
            ruta_analisis = os.path.join(
                self.ruta_base, 
                timestamp, 
                modelo_id, 
                'analisis'
            )
            os.makedirs(ruta_analisis, exist_ok=True)
            
            self.logger.info(f"Guardando análisis para modelo {modelo_id}", {
                'ruta': ruta_analisis,
                'timestamp': timestamp
            })
            
            # 1. Guardar métricas
            self._guardar_metricas(analisis['metricas'], ruta_analisis)
            
            # 2. Guardar gráficos
            self._generar_graficos(analisis, ruta_analisis)
            
            # 3. Guardar análisis detallado
            self._guardar_analisis_detallado(analisis, ruta_analisis)
            
            self.logger.info(f"Análisis guardado exitosamente para modelo {modelo_id}")
            
        except Exception as e:
            self.logger.error(f"Error guardando análisis para modelo {modelo_id}", exc_info=e)
            raise
        
    def _guardar_metricas(self, metricas, ruta):
        """Guarda métricas en formato JSON"""
        try:
            ruta_metricas = os.path.join(ruta, 'metricas.json')
            with open(ruta_metricas, 'w') as f:
                json.dump(metricas, f, indent=4)
            self.logger.debug(f"Métricas guardadas en {ruta_metricas}")
        except Exception as e:
            self.logger.error("Error guardando métricas", exc_info=e)
            raise
            
    def _generar_graficos(self, analisis, ruta):
        """Genera y guarda todos los gráficos"""
        try:
            ruta_graficos = os.path.join(ruta, 'graficos')
            os.makedirs(ruta_graficos, exist_ok=True)
            
            self.logger.info("Generando gráficos de análisis", {
                'ruta': ruta_graficos
            })
            
            # 1. Importancia de variables
            if 'importancia_variables' in analisis:
                self._generar_grafico_importancia(analisis['importancia_variables'], ruta_graficos)
                
            # 2. Distribución de errores
            if 'analisis_errores' in analisis:
                self._generar_grafico_errores(analisis['analisis_errores'], ruta_graficos)
                
            # 3. Tendencias
            if 'tendencias' in analisis:
                self._graficar_tendencias(analisis['tendencias'], ruta_graficos)
                
            self.logger.info("Gráficos generados exitosamente")
            
        except Exception as e:
            self.logger.error("Error generando gráficos", exc_info=e)
            raise
            
    def _generar_grafico_importancia(self, importancia_variables, ruta_graficos):
        """Genera gráfico de importancia de variables"""
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importancia_variables)
            plt.xticks(rotation=45)
            plt.title('Importancia de Variables')
            plt.tight_layout()
            ruta_archivo = os.path.join(ruta_graficos, 'importancia_variables.png')
            plt.savefig(ruta_archivo)
            plt.close()
            self.logger.debug(f"Gráfico de importancia guardado en {ruta_archivo}")
        except Exception as e:
            self.logger.error("Error generando gráfico de importancia", exc_info=e)
            raise
            
    def _generar_grafico_errores(self, analisis_errores, ruta_graficos):
        """Genera gráfico de distribución de errores"""
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(analisis_errores['distribucion'])
            plt.title('Distribución de Errores')
            ruta_archivo = os.path.join(ruta_graficos, 'distribucion_errores.png')
            plt.savefig(ruta_archivo)
            plt.close()
            self.logger.debug(f"Gráfico de errores guardado en {ruta_archivo}")
        except Exception as e:
            self.logger.error("Error generando gráfico de errores", exc_info=e)
            raise
            
    def _graficar_tendencias(self, tendencias, ruta_graficos):
        """Genera gráficos de tendencias"""
        try:
            for nombre, datos in tendencias.items():
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=datos)
                plt.title(f'Tendencia: {nombre}')
                ruta_archivo = os.path.join(ruta_graficos, f'tendencia_{nombre}.png')
                plt.savefig(ruta_archivo)
                plt.close()
                self.logger.debug(f"Gráfico de tendencia {nombre} guardado en {ruta_archivo}")
        except Exception as e:
            self.logger.error("Error generando gráficos de tendencias", exc_info=e)
            raise
            
    def _guardar_analisis_detallado(self, analisis, ruta):
        """Guarda análisis detallado en formato JSON"""
        try:
            ruta_detalle = os.path.join(ruta, 'analisis_detallado.json')
            with open(ruta_detalle, 'w') as f:
                json.dump(analisis, f, indent=4)
            self.logger.debug(f"Análisis detallado guardado en {ruta_detalle}")
        except Exception as e:
            self.logger.error("Error guardando análisis detallado", exc_info=e)
            raise 