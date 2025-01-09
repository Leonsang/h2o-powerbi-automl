import pandas as pd
from .IntegradorH2O_PBI import H2OModeloAvanzado
from .init_h2o_server import iniciar_servidor_h2o, detener_servidor
from .modelo_manager import ModeloManager
from .analizar_resultados import analizar_resultados
from .logger import Logger

logger = Logger('script_pbi')

def ejecutar_prediccion(datos, tipo_modelo='automl', analisis_completo=True):
    """
    Ejecuta el proceso completo de predicción y análisis para Power BI
    
    Args:
        datos: DataFrame con los datos a procesar
        tipo_modelo: Tipo de modelo a utilizar ('automl', 'gbm', etc.)
        analisis_completo: Si True, genera métricas y visualizaciones
    """
    try:
        logger.info("Iniciando proceso de predicción", {
            'tipo_modelo': tipo_modelo,
            'analisis_completo': analisis_completo,
            'shape_datos': datos.shape
        })
        
        # 1. Iniciar servidor
        logger.info("Iniciando servidor H2O")
        servidor_iniciado = iniciar_servidor_h2o()
        if not servidor_iniciado:
            logger.error("No se pudo iniciar el servidor H2O")
            return pd.DataFrame({'Error': ['No se pudo iniciar H2O']})
            
        # 2. Preparar modelo y manager
        logger.info("Preparando modelo y manager")
        modelo = H2OModeloAvanzado()
        manager = ModeloManager()
        
        # 3. Entrenar/predecir
        logger.info("Iniciando entrenamiento/predicción")
        resultado = modelo.entrenar(datos=datos)
        
        # 4. Análisis si es requerido
        if analisis_completo and isinstance(resultado, pd.DataFrame):
            if 'Error' not in resultado.columns:
                logger.info("Realizando análisis completo del modelo")
                analisis = analizar_resultados(
                    modelo=modelo.modelo_actual,
                    datos=datos,
                    predicciones=resultado['prediccion'],
                    tipo_modelo=tipo_modelo
                )
                
                # Agregar información del análisis
                if analisis:
                    logger.debug("Agregando resultados del análisis al DataFrame")
                    for key, value in analisis.items():
                        resultado[f'analisis_{key}'] = str(value)
                        
                # Guardar modelo si es exitoso
                logger.info("Guardando modelo entrenado")
                manager.guardar_modelo(
                    modelo=modelo.modelo_actual,
                    tipo_modelo=tipo_modelo,
                    metricas=analisis.get('metricas', {})
                )
                
        logger.info("Proceso de predicción completado exitosamente")
        return resultado
        
    except Exception as e:
        logger.exception("Error durante el proceso de predicción")
        return pd.DataFrame({'Error': [str(e)]})
        
    finally:
        try:
            logger.info("Deteniendo servidor H2O")
            detener_servidor()
        except Exception as e:
            logger.error("Error al detener servidor H2O", exc_info=e) 