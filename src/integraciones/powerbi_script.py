import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.IntegradorH2O_PBI import H2OModeloAvanzado
from src.analizar_resultados import analizar_resultados
from src.modelo_manager import ModeloManager

def crear_estructura_modelo(nombre_modelo):
    """Crea estructura de directorios para un modelo"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_path = os.path.join("src", "modelos", timestamp, nombre_modelo)
    
    # Crear estructura de directorios
    dirs = {
        'analisis': os.path.join(base_path, 'analisis'),
        'graficos': os.path.join(base_path, 'analisis', 'graficos'),
        'metricas': os.path.join(base_path, 'analisis', 'metricas'),
        'modelo': os.path.join(base_path, 'modelo'),
        'predicciones': os.path.join(base_path, 'predicciones')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def generar_analisis_modelo(datos, predicciones, objetivo, tipo_modelo, dirs):
    """Genera y guarda análisis completo del modelo"""
    # 1. Análisis estadístico
    analisis = analizar_resultados(
        datos=datos,
        predicciones=predicciones,
        objetivo=objetivo,
        tipo_modelo=tipo_modelo
    )
    
    # Guardar métricas
    pd.DataFrame(analisis['metricas']).to_csv(
        os.path.join(dirs['metricas'], 'metricas_modelo.csv')
    )
    
    # 2. Gráficos
    # Predicciones vs Reales
    plt.figure(figsize=(10, 6))
    plt.scatter(datos[objetivo], predicciones, alpha=0.5)
    plt.plot([datos[objetivo].min(), datos[objetivo].max()], 
             [datos[objetivo].min(), datos[objetivo].max()], 'r--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicciones vs Reales - {objetivo}')
    plt.savefig(os.path.join(dirs['graficos'], 'predicciones_vs_reales.png'))
    plt.close()
    
    # Importancia de Variables
    if 'importancia_variables' in analisis:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=analisis['importancia_variables'])
        plt.xticks(rotation=45)
        plt.title(f'Importancia de Variables - {objetivo}')
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['graficos'], 'importancia_variables.png'))
        plt.close()
    
    # Distribución de Errores
    errores = datos[objetivo] - predicciones
    plt.figure(figsize=(10, 6))
    sns.histplot(errores, kde=True)
    plt.title(f'Distribución de Errores - {objetivo}')
    plt.savefig(os.path.join(dirs['graficos'], 'distribucion_errores.png'))
    plt.close()
    
    # 3. Guardar datos de análisis
    analisis_df = pd.DataFrame({
        'real': datos[objetivo],
        'prediccion': predicciones,
        'error': errores
    })
    analisis_df.to_csv(os.path.join(dirs['analisis'], 'analisis_detallado.csv'))
    
    return analisis

def generar_datos_futuros(datos, periodos=12):
    """Genera datos futuros para predicción"""
    n_rows = len(datos)
    datos_futuros = pd.DataFrame()
    
    for columna in datos.columns:
        if pd.api.types.is_numeric_dtype(datos[columna]):
            # Para columnas numéricas, usar tendencia
            tendencia = np.polyfit(range(n_rows), datos[columna], deg=1)
            proyeccion = np.poly1d(tendencia)(range(n_rows, n_rows + periodos))
            datos_futuros[columna] = proyeccion
        else:
            # Para categóricas, usar el último valor
            datos_futuros[columna] = [datos[columna].iloc[-1]] * periodos
    
    # Marcar registros futuros
    datos_futuros['es_prediccion'] = True
    datos_futuros.index = range(n_rows, n_rows + periodos)
    return datos_futuros

def main(dataset, periodos_futuros=12):
    """Integrador entre Power BI y el proyecto H2O"""
    try:
        # 1. Validación inicial de datos
        print("\n=== INICIO DE PROCESAMIENTO ===")
        print(f"• Dataset recibido: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
        print(f"• Columnas disponibles: {dataset.columns.tolist()}")
        
        if dataset is None or len(dataset) == 0:
            print("❌ ERROR: No hay datos para procesar")
            return pd.DataFrame({'Error': ['No hay datos']})
            
        # 2. Inicializar componentes
        print("\n=== INICIALIZACIÓN ===")
        modelo = H2OModeloAvanzado()
        modelo_manager = ModeloManager()
        print("✓ Componentes inicializados")
        
        # 3. Detectar columnas y preparar datos
        print("\n=== ANÁLISIS DE COLUMNAS ===")
        columnas_posibles = modelo.obtener_columnas_posibles(dataset)
        print(f"• Columnas modelables detectadas: {columnas_posibles}")
        
        if not columnas_posibles:
            print("❌ ERROR: No se detectaron columnas modelables")
            return pd.DataFrame({'Error': ['No se detectaron columnas modelables']})
        
        # 4. Preparar datos futuros
        print("\n=== PREPARACIÓN DE DATOS ===")
        datos_futuros = generar_datos_futuros(dataset, periodos_futuros)
        dataset = dataset.copy()
        dataset['es_prediccion'] = False
        datos_futuros['es_prediccion'] = True
        print(f"✓ Datos preparados para predicción futura ({periodos_futuros} periodos)")
        
        # 5. Procesar cada columna modelable
        print("\n=== ENTRENAMIENTO DE MODELOS ===")
        modelos_procesados = 0
        
        for columna in columnas_posibles:
            try:
                print(f"\n• Procesando columna: {columna}")
                
                # Crear estructura
                dirs = crear_estructura_modelo(f"modelo_{columna}")
                print(f"  ✓ Estructura creada en: {dirs['modelo']}")
                
                # Entrenar modelo
                print(f"  • Entrenando modelo para {columna}...")
                resultado = modelo.entrenar(
                    datos=dataset,
                    objetivo=columna
                )
                print(f"  ✓ Modelo entrenado: {resultado['tipo_modelo']}")
                
                # Guardar modelo y generar predicciones
                modelo_manager.guardar_modelo(
                    modelo=resultado['modelo'],
                    nombre=f"modelo_{columna}",
                    ruta=dirs['modelo']
                )
                
                predicciones_actuales = resultado['predicciones']
                predicciones_futuras = modelo.predecir(
                    modelo=resultado['modelo'],
                    datos=datos_futuros
                )
                
                # Guardar predicciones y análisis
                dataset[f'pred_{columna}'] = predicciones_actuales
                datos_futuros[f'pred_{columna}'] = predicciones_futuras
                
                analisis = generar_analisis_modelo(
                    datos=dataset,
                    predicciones=predicciones_actuales,
                    objetivo=columna,
                    tipo_modelo=resultado['tipo_modelo'],
                    dirs=dirs
                )
                
                print(f"  ✓ Análisis y predicciones guardadas")
                modelos_procesados += 1
                
            except Exception as e:
                print(f"  ❌ Error procesando {columna}: {str(e)}")
                continue
        
        # 6. Validar resultados
        print("\n=== RESUMEN FINAL ===")
        print(f"• Modelos procesados: {modelos_procesados} de {len(columnas_posibles)}")
        
        if modelos_procesados == 0:
            print("❌ ERROR: No se pudo procesar ningún modelo")
            return pd.DataFrame({'Error': ['No se pudo procesar ningún modelo']})
        
        # 7. Retornar resultados
        resultado_final = pd.concat([dataset, datos_futuros])
        print(f"✓ Proceso completado. Retornando {resultado_final.shape[0]} filas")
        return resultado_final
        
    except Exception as e:
        print(f"\n❌ ERROR GENERAL: {str(e)}")
        return pd.DataFrame({'Error': [str(e)]}) 