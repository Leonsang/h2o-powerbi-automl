import pandas as pd
import numpy as np
import h2o
import os
import logging
from src.core.feature_engineering import FeatureEngineer
from src.modelos.h2o_modelo import ModeloH2O
from src.mlops.mlops_manager import MLOpsManager
from src.visualizaciones.analisis_resultados import AnalizadorResultados
from src.core.init_h2o_server import iniciar_servidor_h2o, detener_servidor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Inicializar H2O
        logger.info("Iniciando servidor H2O...")
        if not iniciar_servidor_h2o():
            raise RuntimeError("No se pudo iniciar el servidor H2O")
        
        # Cargar datos
        logger.info("Cargando datos...")
        df = pd.read_csv('datos/raw/titanic.csv')
        
        # Preparar datos
        X = df.drop(['Survived'], axis=1)
        y = df['Survived']
        
        # Feature engineering
        logger.info("Aplicando feature engineering...")
        fe = FeatureEngineer()
        X_trans = fe.fit_transform(X, y)
        
        # Crear y entrenar modelo
        logger.info("Entrenando modelo...")
        modelo = ModeloH2O(
            nombre='titanic_classifier',
            tipo='clasificacion',
            config={
                'max_models': 5,
                'max_runtime_secs': 30
            }
        )
        modelo.entrenar(X_trans, y)
        
        # Evaluar modelo
        logger.info("Evaluando modelo...")
        metricas = modelo.evaluar(X_trans, y)
        logger.info(f"Métricas: {metricas}")
        
        # Registrar modelo
        logger.info("Registrando modelo...")
        mlops = MLOpsManager()
        modelo_id = mlops.registrar_modelo(modelo, 'titanic', 'v1')
        logger.info(f"Modelo registrado con ID: {modelo_id}")
        
        # Cargar modelo registrado
        logger.info("Cargando modelo registrado...")
        modelo_cargado = mlops.cargar_modelo(modelo_id)
        
        # Generar predicciones
        logger.info("Generando predicciones...")
        predicciones = modelo_cargado.predecir(X_trans)
        
        # Analizar resultados
        logger.info("Analizando resultados...")
        analizador = AnalizadorResultados()
        reporte = analizador.generar_reporte(
            df=X_trans,
            metricas=metricas,
            modelo_id=modelo_id
        )
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el proceso: {str(e)}")
        raise
    finally:
        # Detener H2O
        detener_servidor()

if __name__ == "__main__":
    main() 