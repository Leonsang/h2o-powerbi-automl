import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

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
    resultados = {}
    
    # 1. Métricas básicas según tipo de modelo
    resultados['metricas'] = calcular_metricas(
        datos[objetivo], 
        predicciones, 
        tipo_modelo
    )
    
    # 2. Análisis de variables importantes
    resultados['importancia_variables'] = analizar_importancia_variables(
        datos, 
        objetivo
    )
    
    # 3. Análisis de errores
    resultados['analisis_errores'] = analizar_errores(
        datos[objetivo], 
        predicciones
    )
    
    # 4. Análisis de segmentos
    resultados['analisis_segmentos'] = analizar_por_segmentos(
        datos, 
        predicciones, 
        objetivo
    )
    
    # 5. Tendencias y patrones
    resultados['tendencias'] = analizar_tendencias(
        datos, 
        predicciones, 
        objetivo
    )
    
    return resultados

def calcular_metricas(reales, predicciones, tipo_modelo):
    """Calcula métricas según tipo de modelo"""
    if tipo_modelo == 'regresion':
        return {
            'r2': r2_score(reales, predicciones),
            'rmse': np.sqrt(mean_squared_error(reales, predicciones)),
            'mae': mean_absolute_error(reales, predicciones),
            'mape': np.mean(np.abs((reales - predicciones) / reales)) * 100
        }
    else:
        return {
            'accuracy': accuracy_score(reales, predicciones),
            'precision': precision_score(reales, predicciones, average='weighted'),
            'recall': recall_score(reales, predicciones, average='weighted'),
            'f1': f1_score(reales, predicciones, average='weighted'),
            'matriz_confusion': confusion_matrix(reales, predicciones).tolist()
        }

def analizar_importancia_variables(datos, objetivo):
    """Analiza importancia de variables mediante correlaciones"""
    correlaciones = datos.corr()[objetivo].sort_values(ascending=False)
    return pd.DataFrame({
        'variable': correlaciones.index,
        'importancia': correlaciones.values
    })

def analizar_errores(reales, predicciones):
    """Análisis detallado de errores"""
    errores = reales - predicciones
    return {
        'distribucion': {
            'mean': errores.mean(),
            'std': errores.std(),
            'skew': errores.skew(),
            'kurtosis': errores.kurtosis()
        },
        'percentiles': {
            '25': np.percentile(errores, 25),
            '50': np.percentile(errores, 50),
            '75': np.percentile(errores, 75)
        }
    }

def analizar_por_segmentos(datos, predicciones, objetivo):
    """Análisis por segmentos de datos"""
    segmentos = {}
    
    # Análisis por cuartiles del objetivo
    cuartiles = pd.qcut(datos[objetivo], q=4)
    for nombre, grupo in datos.groupby(cuartiles):
        idx = grupo.index
        segmentos[f'cuartil_{nombre}'] = {
            'rmse': np.sqrt(mean_squared_error(
                datos.loc[idx, objetivo], 
                predicciones[idx]
            )),
            'size': len(grupo)
        }
    
    return segmentos

def analizar_tendencias(datos, predicciones, objetivo):
    """Análisis de tendencias y patrones"""
    return {
        'tendencia_general': np.polyfit(
            range(len(datos)), 
            datos[objetivo], 
            deg=1
        ).tolist(),
        'estacionalidad': detectar_estacionalidad(datos[objetivo]),
        'outliers': detectar_outliers(datos[objetivo])
    }

def detectar_estacionalidad(serie):
    """Detecta patrones estacionales"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    try:
        descomposicion = seasonal_decompose(
            serie, 
            period=min(len(serie)//2, 12)
        )
        return {
            'seasonal': descomposicion.seasonal.tolist(),
            'trend': descomposicion.trend.tolist(),
            'resid': descomposicion.resid.tolist()
        }
    except:
        return None

def detectar_outliers(serie):
    """Detecta valores atípicos usando IQR"""
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((serie < (Q1 - 1.5 * IQR)) | 
                (serie > (Q3 + 1.5 * IQR)))
    return {
        'indices': outliers[outliers].index.tolist(),
        'valores': serie[outliers].tolist()
    } 