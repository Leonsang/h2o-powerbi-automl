import pandas as pd
from .IntegradorH2O_PBI import H2OModeloAvanzado
from .init_h2o_server import iniciar_servidor_h2o, detener_servidor
from .modelo_manager import ModeloManager
from .analizar_resultados import analizar_resultados

def ejecutar_prediccion(datos, tipo_modelo='automl', analisis_completo=True):
    """
    Ejecuta el proceso completo de predicción y análisis para Power BI
    
    Args:
        datos: DataFrame con los datos a procesar
        tipo_modelo: Tipo de modelo a utilizar ('automl', 'gbm', etc.)
        analisis_completo: Si True, genera métricas y visualizaciones
    """
    try:
        # 1. Iniciar servidor
        servidor_iniciado = iniciar_servidor_h2o()
        if not servidor_iniciado:
            return pd.DataFrame({'Error': ['No se pudo iniciar H2O']})
            
        # 2. Preparar modelo y manager
        modelo = H2OModeloAvanzado()
        manager = ModeloManager()
        
        # 3. Entrenar/predecir
        resultado = modelo.entrenar(datos=datos)
        
        # 4. Análisis si es requerido
        if analisis_completo and isinstance(resultado, pd.DataFrame):
            if 'Error' not in resultado.columns:
                analisis = analizar_resultados(
                    modelo=modelo.modelo_actual,
                    datos=datos,
                    predicciones=resultado['prediccion'],
                    tipo_modelo=tipo_modelo
                )
                
                # Agregar información del análisis
                if analisis:
                    for key, value in analisis.items():
                        resultado[f'analisis_{key}'] = str(value)
                        
                # Guardar modelo si es exitoso
                manager.guardar_modelo(
                    modelo=modelo.modelo_actual,
                    tipo_modelo=tipo_modelo,
                    metricas=analisis.get('metricas', {})
                )
        
        return resultado
        
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})
    finally:
        detener_servidor() 