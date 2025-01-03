import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from .analisis_manager import AnalisisManager

def analizar_resultados(modelo, datos, predicciones, tipo_modelo='automl'):
    """Analiza resultados del modelo y genera reportes"""
    try:
        import seaborn as sns
        usar_seaborn = True
    except ImportError:
        usar_seaborn = False
    
    analisis = AnalisisManager(usar_seaborn=usar_seaborn)
    
    try:
        # Generar reporte básico sin depender de seaborn
        reporte = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'tipo_modelo': tipo_modelo,
            'metricas_basicas': {
                'shape_datos': datos.shape,
                'shape_predicciones': predicciones.shape if hasattr(predicciones, 'shape') else None
            }
        }
        
        # Agregar análisis avanzado si seaborn está disponible
        if usar_seaborn:
            reporte.update(analisis.generar_reporte_completo(modelo, datos, predicciones))
            
        return reporte
        
    except Exception as e:
        print(f"Error analizando resultados: {str(e)}")
        return None

def comparar_modelos(modelos_dict):
    """Compara diferentes modelos y genera análisis comparativo"""
    analisis = AnalisisManager()
    
    try:
        # 1. Recopilar métricas de todos los modelos
        metricas_comparativas = {}
        for nombre, modelo in modelos_dict.items():
            metricas = analisis._calcular_metricas_rendimiento(modelo)
            metricas_comparativas[nombre] = metricas
        
        # 2. Crear DataFrame comparativo
        df_comparacion = pd.DataFrame(metricas_comparativas).T
        
        # 3. Generar gráfico comparativo
        fig, ax = plt.subplots(figsize=(12, 6))
        df_comparacion.plot(kind='bar', ax=ax)
        plt.title('Comparación de Modelos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 4. Guardar resultados
        analisis.guardar_metricas(
            df_comparacion, 
            'comparativa', 
            'rendimiento'
        )
        analisis.guardar_grafico(
            fig, 
            'comparativa', 
            'comparacion_modelos'
        )
        
        return df_comparacion
        
    except Exception as e:
        print(f"Error comparando modelos: {str(e)}")
        return None 