import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import os
import json
from datetime import datetime
import logging
from .visualizador import Visualizador

logger = logging.getLogger(__name__)

class AnalizadorResultados:
    """Clase para analizar resultados de modelos"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el analizador
        
        Args:
            config: Configuración del analizador
        """
        self.config = config or {}
        self.visualizador = Visualizador(config)
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output', 'reportes'))
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analizar_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza un dataset
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con resultados del análisis
        """
        try:
            analisis = {
                'registros': len(df),
                'variables': len(df.columns),
                'tipos_variables': df.dtypes.value_counts().to_dict(),
                'valores_nulos': df.isnull().sum().to_dict(),
                'estadisticas': self._obtener_estadisticas(df),
                'correlaciones': self._obtener_correlaciones(df)
            }
            
            return analisis
            
        except Exception as e:
            logger.error(f"Error analizando dataset: {str(e)}")
            raise
    
    def analizar_modelo(self, metricas: Dict[str, Any], importancias: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analiza resultados de un modelo
        
        Args:
            metricas: Métricas del modelo
            importancias: Importancia de variables (opcional)
            
        Returns:
            Diccionario con análisis del modelo
        """
        try:
            analisis = {
                'metricas_principales': self._obtener_metricas_principales(metricas),
                'interpretacion': self._interpretar_metricas(metricas)
            }
            
            if importancias:
                analisis['variables_importantes'] = self._analizar_importancia_variables(importancias)
            
            return analisis
            
        except Exception as e:
            logger.error(f"Error analizando modelo: {str(e)}")
            raise
    
    def generar_reporte(self, df: pd.DataFrame, metricas: Dict[str, Any], 
                       importancias: Optional[Dict[str, float]] = None,
                       modelo_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera un reporte completo
        
        Args:
            df: DataFrame con los datos
            metricas: Métricas del modelo
            importancias: Importancia de variables (opcional)
            modelo_id: ID del modelo (opcional)
            
        Returns:
            Diccionario con el reporte completo
        """
        try:
            # Analizar datos y modelo
            analisis_datos = self.analizar_dataset(df)
            analisis_modelo = self.analizar_modelo(metricas, importancias)
            
            # Generar visualizaciones
            visualizaciones = self.visualizador.generar_reporte_visual(
                df, metricas, importancias, modelo_id
            )
            
            # Crear reporte
            reporte = {
                'fecha_reporte': datetime.now().isoformat(),
                'modelo_id': modelo_id,
                'analisis_datos': analisis_datos,
                'analisis_modelo': analisis_modelo,
                'visualizaciones': visualizaciones,
                'recomendaciones': self._generar_recomendaciones(analisis_datos, analisis_modelo)
            }
            
            # Guardar reporte
            if modelo_id:
                ruta_reporte = os.path.join(self.output_dir, f"reporte_{modelo_id}.json")
                with open(ruta_reporte, 'w', encoding='utf-8') as f:
                    json.dump(reporte, f, indent=4, ensure_ascii=False)
            
            return reporte
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            raise
    
    def _obtener_estadisticas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtiene estadísticas descriptivas del dataset
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con estadísticas
        """
        try:
            # Estadísticas para variables numéricas
            numericas = df.select_dtypes(include=[np.number])
            stats_num = numericas.describe().to_dict()
            
            # Estadísticas para variables categóricas
            categoricas = df.select_dtypes(exclude=[np.number])
            stats_cat = {col: df[col].value_counts().to_dict() for col in categoricas.columns}
            
            return {
                'numericas': stats_num,
                'categoricas': stats_cat
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            raise
    
    def _obtener_correlaciones(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtiene correlaciones entre variables numéricas
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con correlaciones
        """
        try:
            numericas = df.select_dtypes(include=[np.number])
            corr = numericas.corr()
            
            # Encontrar correlaciones más fuertes
            correlaciones_fuertes = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i,j]) > 0.7:
                        correlaciones_fuertes.append({
                            'variables': [corr.columns[i], corr.columns[j]],
                            'correlacion': corr.iloc[i,j]
                        })
            
            return {
                'matriz': corr.to_dict(),
                'correlaciones_fuertes': correlaciones_fuertes
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo correlaciones: {str(e)}")
            raise
    
    def _obtener_metricas_principales(self, metricas: Dict[str, Any]) -> Dict[str, float]:
        """
        Obtiene las métricas principales del modelo
        
        Args:
            metricas: Diccionario con métricas
            
        Returns:
            Diccionario con métricas principales
        """
        try:
            principales = {}
            
            # Métricas comunes
            if 'rmse' in metricas:
                principales['rmse'] = metricas['rmse']
            if 'r2' in metricas:
                principales['r2'] = metricas['r2']
            
            # Métricas de clasificación
            if 'auc' in metricas:
                principales['auc'] = metricas['auc']
            if 'precision' in metricas:
                principales['precision'] = metricas['precision']
            if 'recall' in metricas:
                principales['recall'] = metricas['recall']
            if 'f1' in metricas:
                principales['f1'] = metricas['f1']
            
            return principales
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas principales: {str(e)}")
            raise
    
    def _interpretar_metricas(self, metricas: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpreta las métricas del modelo
        
        Args:
            metricas: Diccionario con métricas
            
        Returns:
            Diccionario con interpretaciones
        """
        try:
            interpretacion = {}
            
            # Interpretar RMSE
            if 'rmse' in metricas:
                rmse = metricas['rmse']
                if rmse < 0.1:
                    interpretacion['rmse'] = "Error muy bajo, excelente ajuste"
                elif rmse < 0.3:
                    interpretacion['rmse'] = "Error bajo, buen ajuste"
                else:
                    interpretacion['rmse'] = "Error alto, ajuste mejorable"
            
            # Interpretar R2
            if 'r2' in metricas:
                r2 = metricas['r2']
                if r2 > 0.9:
                    interpretacion['r2'] = "Ajuste excelente"
                elif r2 > 0.7:
                    interpretacion['r2'] = "Buen ajuste"
                else:
                    interpretacion['r2'] = "Ajuste mejorable"
            
            # Interpretar AUC
            if 'auc' in metricas:
                auc = metricas['auc']
                if auc > 0.9:
                    interpretacion['auc'] = "Discriminación excelente"
                elif auc > 0.7:
                    interpretacion['auc'] = "Buena discriminación"
                else:
                    interpretacion['auc'] = "Discriminación mejorable"
            
            return interpretacion
            
        except Exception as e:
            logger.error(f"Error interpretando métricas: {str(e)}")
            raise
    
    def _analizar_importancia_variables(self, importancias: Dict[str, float]) -> Dict[str, Any]:
        """
        Analiza la importancia de las variables
        
        Args:
            importancias: Diccionario con importancias
            
        Returns:
            Diccionario con análisis de importancia
        """
        try:
            # Ordenar variables por importancia
            importancias_sorted = dict(sorted(importancias.items(), key=lambda x: x[1], reverse=True))
            
            # Identificar variables más importantes (top 5)
            top_vars = list(importancias_sorted.items())[:5]
            
            # Calcular importancia relativa
            total_imp = sum(importancias.values())
            importancia_relativa = {var: imp/total_imp for var, imp in importancias.items()}
            
            return {
                'top_variables': top_vars,
                'importancia_relativa': importancia_relativa
            }
            
        except Exception as e:
            logger.error(f"Error analizando importancia de variables: {str(e)}")
            raise
    
    def _generar_recomendaciones(self, analisis_datos: Dict[str, Any], 
                               analisis_modelo: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones basadas en los análisis
        
        Args:
            analisis_datos: Análisis del dataset
            analisis_modelo: Análisis del modelo
            
        Returns:
            Lista de recomendaciones
        """
        try:
            recomendaciones = []
            
            # Recomendaciones basadas en datos
            if analisis_datos['valores_nulos']:
                recomendaciones.append(
                    "Hay valores nulos en los datos. Considerar técnicas de imputación."
                )
            
            if analisis_datos['correlaciones']['correlaciones_fuertes']:
                recomendaciones.append(
                    "Hay variables altamente correlacionadas. Considerar reducción de dimensionalidad."
                )
            
            # Recomendaciones basadas en modelo
            metricas = analisis_modelo['metricas_principales']
            
            if 'rmse' in metricas and metricas['rmse'] > 0.3:
                recomendaciones.append(
                    "El error (RMSE) es alto. Considerar feature engineering o probar otros algoritmos."
                )
            
            if 'r2' in metricas and metricas['r2'] < 0.7:
                recomendaciones.append(
                    "El R2 es bajo. Considerar incluir más variables o transformar las existentes."
                )
            
            if 'auc' in metricas and metricas['auc'] < 0.7:
                recomendaciones.append(
                    "El AUC es bajo. Considerar balancear las clases o ajustar hiperparámetros."
                )
            
            return recomendaciones
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {str(e)}")
            raise 