import h2o
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import os
import json
from datetime import datetime
import logging
from ..core.modelo_base import ModeloBase

logger = logging.getLogger(__name__)

class ModeloH2O(ModeloBase):
    """Implementación de modelo usando H2O AutoML"""
    
    def __init__(self, nombre: str, tipo: str = 'clasificacion', config: Optional[Dict[str, Any]] = None):
        """
        Inicializa un modelo H2O
        
        Args:
            nombre: Nombre del modelo
            tipo: Tipo de modelo (clasificacion/regresion)
            config: Configuración del modelo
        """
        super().__init__(nombre, tipo, config)
        self.modelo = None
        self.configuracion_default = {
            'max_models': 10,
            'seed': 42,
            'max_runtime_secs': 300,
            'include_algos': ["DRF", "GBM", "GLM"],
            'nfolds': 5,
            'keep_cross_validation_predictions': True,
            'sort_metric': 'AUTO'
        }
        self.config = {**self.configuracion_default, **(config or {})}
        
    def entrenar(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Entrena el modelo H2O con AutoML
        
        Args:
            X: Features de entrenamiento
            y: Variable objetivo
        """
        try:
            if not self.validar_datos(X, y):
                raise ValueError("Datos inválidos para entrenamiento")
            
            # Convertir a H2O Frame
            df = X.copy()
            df[y.name] = y
            h2o_frame = h2o.H2OFrame(df)
            
            # Convertir variable objetivo a factor si es clasificación
            if self.tipo == 'clasificacion':
                h2o_frame[y.name] = h2o_frame[y.name].asfactor()
            
            # Particionar datos
            train, valid, test = h2o_frame.split_frame([0.7, 0.15])
            
            # Configurar AutoML
            aml = h2o.automl.H2OAutoML(
                max_models=self.config['max_models'],
                seed=self.config['seed'],
                max_runtime_secs=self.config['max_runtime_secs'],
                include_algos=self.config['include_algos'],
                nfolds=self.config['nfolds'],
                keep_cross_validation_predictions=self.config['keep_cross_validation_predictions'],
                sort_metric=self.config['sort_metric']
            )
            
            # Entrenar
            aml.train(y=y.name, training_frame=train, validation_frame=valid, leaderboard_frame=test)
            
            # Guardar modelo y métricas
            self.modelo = aml.leader
            self.metricas = self._obtener_metricas_h2o(test)
            self.esta_entrenado = True
            
            logger.info(f"Modelo {self.nombre} entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {str(e)}")
            raise
    
    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el modelo H2O
        
        Args:
            X: Datos para predecir
            
        Returns:
            Array con predicciones
        """
        try:
            if not self.esta_entrenado:
                raise ValueError("El modelo no está entrenado")
            
            if not self.validar_datos(X):
                raise ValueError("Datos inválidos para predicción")
            
            # Convertir a H2O Frame
            h2o_frame = h2o.H2OFrame(X)
            
            # Predecir
            predicciones = self.modelo.predict(h2o_frame)
            
            return predicciones.as_data_frame().values
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise
    
    def evaluar(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo H2O
        
        Args:
            X: Datos de evaluación
            y: Valores reales
            
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            if not self.esta_entrenado:
                raise ValueError("El modelo no está entrenado")
            
            # Convertir a H2O Frame
            df = X.copy()
            df[y.name] = y
            h2o_frame = h2o.H2OFrame(df)
            
            # Evaluar
            return self._obtener_metricas_h2o(h2o_frame)
            
        except Exception as e:
            logger.error(f"Error en evaluación: {str(e)}")
            raise
    
    def guardar(self, ruta: str) -> None:
        """
        Guarda el modelo H2O en disco
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        try:
            if not self.esta_entrenado:
                raise ValueError("El modelo no está entrenado")
            
            # Crear directorio si no existe
            os.makedirs(ruta, exist_ok=True)
            
            # Guardar modelo H2O
            modelo_path = h2o.save_model(self.modelo, path=ruta, force=True)
            
            # Guardar metadata
            metadata = self.generar_metadata()
            metadata['ruta_modelo'] = modelo_path
            
            with open(os.path.join(ruta, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Modelo guardado en: {ruta}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            raise
    
    def cargar(self, ruta: str) -> None:
        """
        Carga el modelo H2O desde disco
        
        Args:
            ruta: Ruta del modelo guardado
        """
        try:
            # Verificar que existe el directorio
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"No se encontró el directorio: {ruta}")
            
            # Cargar metadata
            metadata_path = os.path.join(ruta, 'metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Cargar modelo H2O
            self.modelo = h2o.load_model(metadata['ruta_modelo'])
            self.metricas = metadata['metricas']
            self.esta_entrenado = True
            
            logger.info(f"Modelo cargado desde: {ruta}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def _obtener_metricas_h2o(self, datos: h2o.H2OFrame) -> Dict[str, float]:
        """
        Obtiene métricas específicas de H2O
        
        Args:
            datos: H2OFrame para evaluar
            
        Returns:
            Diccionario con métricas
        """
        try:
            perf = self.modelo.model_performance(datos)
            
            metricas = {
                'rmse': perf.rmse(),
                'mse': perf.mse(),
                'r2': perf.r2()
            }
            
            # Métricas específicas para clasificación
            if self.tipo == 'clasificacion':
                metricas.update({
                    'auc': perf.auc() if hasattr(perf, 'auc') else None,
                    'precision': perf.precision() if hasattr(perf, 'precision') else None,
                    'recall': perf.recall() if hasattr(perf, 'recall') else None,
                    'f1': perf.F1() if hasattr(perf, 'F1') else None,
                    'confusion_matrix': perf.confusion_matrix().table.as_data_frame().to_dict() if hasattr(perf, 'confusion_matrix') else None
                })
            
            return metricas
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas H2O: {str(e)}")
            raise 