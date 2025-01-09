from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ModeloBase(ABC):
    """Clase base abstracta para todos los modelos"""
    
    def __init__(self, nombre: str, tipo: str = 'clasificacion', config: Optional[Dict[str, Any]] = None):
        """
        Inicializa un modelo base
        
        Args:
            nombre: Nombre del modelo
            tipo: Tipo de modelo (clasificacion/regresion)
            config: Configuración del modelo
        """
        self.nombre = nombre
        self.tipo = tipo
        self.config = config or {}
        self.esta_entrenado = False
        self.metricas = {}
        self.fecha_creacion = datetime.now()
        self.fecha_ultimo_entrenamiento = None
        
    @abstractmethod
    def entrenar(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Entrena el modelo con los datos proporcionados
        
        Args:
            X: Features de entrenamiento
            y: Variable objetivo
        """
        pass
    
    @abstractmethod
    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el modelo
        
        Args:
            X: Datos para predecir
            
        Returns:
            Array con predicciones
        """
        pass
    
    @abstractmethod
    def evaluar(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo
        
        Args:
            X: Datos de evaluación
            y: Valores reales
            
        Returns:
            Diccionario con métricas de evaluación
        """
        pass
    
    @abstractmethod
    def guardar(self, ruta: str) -> None:
        """
        Guarda el modelo en disco
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        pass
    
    @abstractmethod
    def cargar(self, ruta: str) -> None:
        """
        Carga el modelo desde disco
        
        Args:
            ruta: Ruta del modelo guardado
        """
        pass
    
    def validar_datos(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
        """
        Valida los datos de entrada
        
        Args:
            X: Features a validar
            y: Variable objetivo a validar (opcional)
            
        Returns:
            True si los datos son válidos
        """
        try:
            # Validar que X es un DataFrame
            if not isinstance(X, pd.DataFrame):
                logger.error("X debe ser un DataFrame")
                return False
            
            # Validar que no hay valores nulos
            if X.isnull().any().any():
                logger.warning("X contiene valores nulos")
            
            # Si se proporciona y, validar
            if y is not None:
                if not isinstance(y, pd.Series):
                    logger.error("y debe ser una Series")
                    return False
                
                if y.isnull().any():
                    logger.warning("y contiene valores nulos")
                
                if len(X) != len(y):
                    logger.error("X e y deben tener la misma longitud")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando datos: {str(e)}")
            return False
    
    def generar_metadata(self) -> Dict[str, Any]:
        """
        Genera metadata del modelo
        
        Returns:
            Diccionario con metadata
        """
        return {
            'nombre': self.nombre,
            'tipo': self.tipo,
            'config': self.config,
            'esta_entrenado': self.esta_entrenado,
            'metricas': self.metricas,
            'fecha_creacion': self.fecha_creacion.isoformat(),
            'fecha_ultimo_entrenamiento': self.fecha_ultimo_entrenamiento.isoformat() if self.fecha_ultimo_entrenamiento else None
        } 