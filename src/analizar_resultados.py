import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from .logger import Logger

logger = Logger('analizar_resultados')

def analizar_resultados(datos, predicciones, objetivo, tipo_modelo):
    """
    Análisis completo de resultados del modelo
    
    Args:
        datos: DataFrame original
        predicciones: Series con predicciones
        objetivo: Nombre de columna objetivo
        tipo_modelo: 'clasificacion' o 'regresion'
        
    Returns:
        dict con análisis completo
    """
    try:
        logger.info("Iniciando análisis de resultados", {
            'tipo_modelo': tipo_modelo,
            'objetivo': objetivo,
            'shape_datos': datos.shape
        })
        
        resultados = {}
        
        # 1. Métricas básicas según tipo de modelo
        logger.info("Calculando métricas básicas")
        resultados['metricas'] = calcular_metricas(
            datos[objetivo], 
            predicciones, 
            tipo_modelo
        )
        
        # 2. Análisis de variables importantes
        logger.info("Analizando importancia de variables")
        resultados['importancia_variables'] = analizar_importancia_variables(
            datos, 
            objetivo
        )
        
        # 3. Análisis de errores
        logger.info("Analizando errores")
        resultados['analisis_errores'] = analizar_errores(
            datos[objetivo], 
            predicciones
        )
        
        # 4. Análisis de segmentos
        logger.info("Analizando segmentos")
        resultados['analisis_segmentos'] = analizar_por_segmentos(
            datos, 
            predicciones, 
            objetivo
        )
        
        # 5. Tendencias y patrones
        logger.info("Analizando tendencias y patrones")
        resultados['tendencias'] = analizar_tendencias(
            datos, 
            predicciones, 
            objetivo
        )
        
        logger.info("Análisis de resultados completado exitosamente")
        return resultados
        
    except Exception as e:
        logger.exception("Error durante el análisis de resultados")
        raise

def calcular_metricas(reales, predicciones, tipo_modelo):
    """Calcula métricas según tipo de modelo"""
    try:
        logger.debug("Calculando métricas", {'tipo_modelo': tipo_modelo})
        
        if tipo_modelo == 'clasificacion':
            metricas = {
                'accuracy': accuracy_score(reales, predicciones),
                'precision': precision_score(reales, predicciones, average='weighted'),
                'recall': recall_score(reales, predicciones, average='weighted'),
                'f1': f1_score(reales, predicciones, average='weighted'),
                'confusion_matrix': confusion_matrix(reales, predicciones).tolist()
            }
        else:
            metricas = {
                'r2': r2_score(reales, predicciones),
                'rmse': np.sqrt(mean_squared_error(reales, predicciones)),
                'mae': mean_absolute_error(reales, predicciones)
            }
            
        logger.debug("Métricas calculadas", metricas)
        return metricas
        
    except Exception as e:
        logger.error("Error calculando métricas", exc_info=e)
        raise

def analizar_importancia_variables(datos, objetivo):
    """Analiza importancia de variables"""
    try:
        logger.debug("Analizando importancia de variables")
        # Implementar análisis de importancia
        return {}
        
    except Exception as e:
        logger.error("Error analizando importancia de variables", exc_info=e)
        raise

def analizar_errores(reales, predicciones):
    """Analiza distribución y patrones de errores"""
    try:
        logger.debug("Analizando errores")
        errores = np.abs(reales - predicciones)
        
        analisis = {
            'error_medio': errores.mean(),
            'error_std': errores.std(),
            'error_max': errores.max(),
            'error_min': errores.min(),
            'distribucion': errores.tolist()
        }
        
        logger.debug("Análisis de errores completado", analisis)
        return analisis
        
    except Exception as e:
        logger.error("Error analizando errores", exc_info=e)
        raise

def analizar_por_segmentos(datos, predicciones, objetivo):
    """Analiza comportamiento por segmentos"""
    try:
        logger.debug("Analizando por segmentos")
        # Implementar análisis por segmentos
        return {}
        
    except Exception as e:
        logger.error("Error analizando segmentos", exc_info=e)
        raise

def analizar_tendencias(datos, predicciones, objetivo):
    """Analiza tendencias y patrones temporales"""
    try:
        logger.debug("Analizando tendencias")
        # Implementar análisis de tendencias
        return {}
        
    except Exception as e:
        logger.error("Error analizando tendencias", exc_info=e)
        raise 