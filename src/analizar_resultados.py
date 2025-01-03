import pandas as pd
import numpy as np
from .logger import Logger

def analizar_resultados(modelo, datos, predicciones, tipo_modelo='automl'):
    """
    Analiza resultados del modelo y genera métricas e insights.
    
    Args:
        modelo: Modelo H2O entrenado
        datos: DataFrame original
        predicciones: Predicciones del modelo
        tipo_modelo: Tipo de modelo usado
        
    Returns:
        Dict con análisis completo
    """
    logger = Logger('analisis_resultados')
    
    try:
        # 1. Métricas básicas
        metricas_basicas = {
            'r2': modelo.r2(),
            'rmse': modelo.rmse(),
            'mae': modelo.mae()
        }
        
        # 2. Análisis de residuos
        residuos = None
        if hasattr(predicciones, 'values'):
            residuos = datos.values - predicciones.values
        
        # 3. Importancia de variables
        importancia = None
        if hasattr(modelo, 'varimp'):
            importancia = modelo.varimp(use_pandas=True)
        
        # 4. Estadísticas de predicciones
        stats_predicciones = {
            'min': float(np.min(predicciones)),
            'max': float(np.max(predicciones)),
            'mean': float(np.mean(predicciones)),
            'std': float(np.std(predicciones))
        }
        
        # 5. Generar reporte
        reporte = {
            'metricas_basicas': metricas_basicas,
            'estadisticas_predicciones': stats_predicciones,
            'tipo_modelo': tipo_modelo,
            'num_registros': len(datos),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if residuos is not None:
            reporte['analisis_residuos'] = {
                'mean': float(np.mean(residuos)),
                'std': float(np.std(residuos))
            }
            
        if importancia is not None:
            reporte['importancia_variables'] = importancia.to_dict()
            
        logger.info("Análisis de resultados completado")
        return reporte
        
    except Exception as e:
        logger.error(f"Error en análisis de resultados: {str(e)}")
        raise 