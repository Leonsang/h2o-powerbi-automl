import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Clase para ingeniería de características"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el feature engineer
        
        Args:
            config: Configuración del feature engineer
        """
        self.config = config or {}
        self.transformadores = {}
        self.columnas_originales = None
        self.columnas_generadas = None
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta las transformaciones y transforma los datos
        
        Args:
            df: DataFrame a transformar
            y: Variable objetivo (opcional)
            
        Returns:
            DataFrame transformado
        """
        try:
            # Guardar columnas originales
            self.columnas_originales = df.columns.tolist()
            
            # Aplicar transformaciones
            df_trans = df.copy()
            
            # 1. Manejo de valores nulos
            df_trans = self._manejar_nulos(df_trans, fit=True)
            
            # 2. Codificación de variables categóricas
            df_trans = self._codificar_categoricas(df_trans, fit=True)
            
            # 3. Escalado de variables numéricas
            df_trans = self._escalar_numericas(df_trans, fit=True)
            
            # 4. Generación de características
            df_trans = self._generar_caracteristicas(df_trans, y, fit=True)
            
            # Guardar columnas generadas
            self.columnas_generadas = df_trans.columns.tolist()
            
            return df_trans
            
        except Exception as e:
            logger.error(f"Error en fit_transform: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma nuevos datos usando las transformaciones ajustadas
        
        Args:
            df: DataFrame a transformar
            
        Returns:
            DataFrame transformado
        """
        try:
            if not self.transformadores:
                raise ValueError("El feature engineer no ha sido ajustado")
            
            df_trans = df.copy()
            
            # 1. Manejo de valores nulos
            df_trans = self._manejar_nulos(df_trans, fit=False)
            
            # 2. Codificación de variables categóricas
            df_trans = self._codificar_categoricas(df_trans, fit=False)
            
            # 3. Escalado de variables numéricas
            df_trans = self._escalar_numericas(df_trans, fit=False)
            
            # 4. Generación de características
            df_trans = self._generar_caracteristicas(df_trans, None, fit=False)
            
            # Asegurar que tenemos todas las columnas necesarias
            for col in self.columnas_generadas:
                if col not in df_trans.columns:
                    df_trans[col] = 0
            
            # Ordenar columnas como en el entrenamiento
            df_trans = df_trans[self.columnas_generadas]
            
            return df_trans
            
        except Exception as e:
            logger.error(f"Error en transform: {str(e)}")
            raise
    
    def _manejar_nulos(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Maneja valores nulos en el dataset
        
        Args:
            df: DataFrame a procesar
            fit: Si es True, ajusta los imputadores
            
        Returns:
            DataFrame sin valores nulos
        """
        try:
            df_clean = df.copy()
            
            # Separar variables numéricas y categóricas
            numericas = df.select_dtypes(include=[np.number]).columns
            categoricas = df.select_dtypes(exclude=[np.number]).columns
            
            # Imputar numéricas
            if numericas.any():
                if fit:
                    self.transformadores['imputer_num'] = SimpleImputer(
                        strategy='mean'
                    )
                    self.transformadores['imputer_num'].fit(df[numericas])
                
                df_clean[numericas] = self.transformadores['imputer_num'].transform(
                    df[numericas]
                )
            
            # Imputar categóricas
            if categoricas.any():
                if fit:
                    self.transformadores['imputer_cat'] = SimpleImputer(
                        strategy='most_frequent'
                    )
                    self.transformadores['imputer_cat'].fit(df[categoricas])
                
                df_clean[categoricas] = self.transformadores['imputer_cat'].transform(
                    df[categoricas]
                )
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error manejando nulos: {str(e)}")
            raise
    
    def _codificar_categoricas(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Codifica variables categóricas
        
        Args:
            df: DataFrame a procesar
            fit: Si es True, ajusta los codificadores
            
        Returns:
            DataFrame con variables codificadas
        """
        try:
            df_coded = df.copy()
            
            # Identificar variables categóricas
            categoricas = df.select_dtypes(exclude=[np.number]).columns
            
            if not categoricas.empty:
                if fit:
                    self.transformadores['encoders'] = {}
                
                for col in categoricas:
                    if fit:
                        self.transformadores['encoders'][col] = LabelEncoder()
                        self.transformadores['encoders'][col].fit(df[col].astype(str))
                    
                    df_coded[col] = self.transformadores['encoders'][col].transform(
                        df[col].astype(str)
                    )
            
            return df_coded
            
        except Exception as e:
            logger.error(f"Error codificando categóricas: {str(e)}")
            raise
    
    def _escalar_numericas(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Escala variables numéricas
        
        Args:
            df: DataFrame a procesar
            fit: Si es True, ajusta los escaladores
            
        Returns:
            DataFrame con variables escaladas
        """
        try:
            df_scaled = df.copy()
            
            # Identificar variables numéricas
            numericas = df.select_dtypes(include=[np.number]).columns
            
            if not numericas.empty:
                if fit:
                    self.transformadores['scaler'] = StandardScaler()
                    self.transformadores['scaler'].fit(df[numericas])
                
                df_scaled[numericas] = self.transformadores['scaler'].transform(
                    df[numericas]
                )
            
            return df_scaled
            
        except Exception as e:
            logger.error(f"Error escalando numéricas: {str(e)}")
            raise
    
    def _generar_caracteristicas(self, df: pd.DataFrame, y: Optional[pd.Series] = None,
                               fit: bool = True) -> pd.DataFrame:
        """
        Genera nuevas características
        
        Args:
            df: DataFrame a procesar
            y: Variable objetivo (opcional)
            fit: Si es True, genera nuevas características
            
        Returns:
            DataFrame con nuevas características
        """
        try:
            df_features = df.copy()
            
            # Identificar variables numéricas
            numericas = df.select_dtypes(include=[np.number]).columns
            
            if fit:
                # 1. Interacciones entre variables numéricas
                for i in range(len(numericas)):
                    for j in range(i+1, len(numericas)):
                        col1, col2 = numericas[i], numericas[j]
                        # Multiplicación
                        df_features[f'interaccion_{col1}_{col2}'] = df[col1] * df[col2]
                        # Ratio (evitando división por cero)
                        df_features[f'ratio_{col1}_{col2}'] = df[col1] / (df[col2] + 1e-6)
                
                # 2. Características polinómicas
                for col in numericas:
                    df_features[f'{col}_cuadrado'] = df[col] ** 2
                    df_features[f'{col}_cubo'] = df[col] ** 3
                
                # 3. Características con la variable objetivo (si se proporciona)
                if y is not None:
                    target_mean = df.groupby(y).mean()
                    for col in numericas:
                        df_features[f'target_mean_{col}'] = df[col] - target_mean[col].mean()
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error generando características: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Obtiene los nombres de las características generadas
        
        Returns:
            Lista con nombres de características
        """
        if self.columnas_generadas is None:
            raise ValueError("El feature engineer no ha sido ajustado")
        return self.columnas_generadas
    
    def get_original_features(self) -> List[str]:
        """
        Obtiene los nombres de las características originales
        
        Returns:
            Lista con nombres de características originales
        """
        if self.columnas_originales is None:
            raise ValueError("El feature engineer no ha sido ajustado")
        return self.columnas_originales 