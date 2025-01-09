import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class OutlierDetector:
    def __init__(self, method='zscore', threshold=3):
        self.method = method
        self.threshold = threshold
    
    def process(self, data):
        """Detecta y procesa outliers en datos numéricos"""
        df = data.copy()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if self.method == 'zscore':
            for col in numeric_cols:
                if df[col].notnull().any():
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df.loc[z_scores > self.threshold, col] = df[col].median()
        
        return df

class DatasetAnalyzer:
    """Analizador automático de datasets para determinar estrategias de feature engineering"""
    
    def __init__(self):
        self.analisis = {}
    
    def analizar_dataset(self, df: pd.DataFrame) -> dict:
        """Analiza el dataset y recomienda estrategias de feature engineering"""
        try:
            analisis = {
                'estructura': self._analizar_estructura(df),
                'tipos_datos': self._analizar_tipos_datos(df),
                'calidad_datos': self._analizar_calidad_datos(df),
                'patrones': self._analizar_patrones(df),
                'recomendaciones': []
            }
            
            # Generar recomendaciones basadas en el análisis
            self._generar_recomendaciones(analisis)
            
            return analisis
        except Exception as e:
            logger.error(f"Error analizando dataset: {str(e)}")
            return {}
    
    def _analizar_estructura(self, df: pd.DataFrame) -> dict:
        """Analiza la estructura básica del dataset"""
        return {
            'n_registros': len(df),
            'n_features': len(df.columns),
            'memoria_uso': df.memory_usage().sum() / 1024**2,  # MB
            'sparsity': df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        }
    
    def _analizar_tipos_datos(self, df: pd.DataFrame) -> dict:
        """Analiza los tipos de datos presentes"""
        tipos = {
            'numericas': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categoricas': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'temporales': [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])],
            'binarias': [col for col in df.columns if set(df[col].dropna().unique()) <= {0, 1}],
            'alta_cardinalidad': [col for col in df.select_dtypes(include=['object']) 
                                if df[col].nunique() / len(df) > 0.5]
        }
        return tipos
    
    def _analizar_calidad_datos(self, df: pd.DataFrame) -> dict:
        """Analiza la calidad de los datos"""
        return {
            'valores_nulos': df.isnull().sum().to_dict(),
            'valores_unicos': df.nunique().to_dict(),
            'outliers': self._detectar_outliers(df),
            'correlaciones': self._analizar_correlaciones(df)
        }
    
    def _analizar_patrones(self, df: pd.DataFrame) -> dict:
        """Analiza patrones en los datos"""
        patrones = {
            'ciclicidad': {},
            'tendencias': {},
            'clusters_naturales': {}
        }
        
        # Detectar patrones temporales
        fechas = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if fechas:
            for col in fechas:
                patrones['ciclicidad'][col] = self._detectar_ciclicidad(df[col])
        
        # Detectar tendencias en numéricas
        numericas = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numericas:
            if df[col].notnull().any():
                patrones['tendencias'][col] = self._detectar_tendencia(df[col])
        
        return patrones
    
    def _detectar_outliers(self, df: pd.DataFrame) -> dict:
        """Detecta outliers en variables numéricas"""
        outliers = {}
        for col in df.select_dtypes(include=['int64', 'float64']):
            if df[col].notnull().any():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = {
                    'n_outliers': len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]),
                    'porcentaje': len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]) / len(df)
                }
        return outliers
    
    def _analizar_correlaciones(self, df: pd.DataFrame) -> dict:
        """Analiza correlaciones entre variables numéricas"""
        numericas = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numericas) > 1:
            corr_matrix = df[numericas].corr()
            correlaciones_altas = {}
            for i in range(len(numericas)):
                for j in range(i+1, len(numericas)):
                    corr = corr_matrix.iloc[i,j]
                    if abs(corr) > 0.7:
                        correlaciones_altas[f"{numericas[i]}_{numericas[j]}"] = corr
            return correlaciones_altas
        return {}
    
    def _detectar_ciclicidad(self, serie: pd.Series) -> dict:
        """Detecta patrones cíclicos en series temporales"""
        if pd.api.types.is_datetime64_any_dtype(serie):
            return {
                'diario': serie.dt.hour.value_counts().index[0] if hasattr(serie.dt, 'hour') else None,
                'semanal': serie.dt.dayofweek.value_counts().index[0] if hasattr(serie.dt, 'dayofweek') else None,
                'mensual': serie.dt.day.value_counts().index[0] if hasattr(serie.dt, 'day') else None
            }
        return {}
    
    def _detectar_tendencia(self, serie: pd.Series) -> dict:
        """Detecta tendencias en series numéricas"""
        if len(serie.dropna()) > 1:
            try:
                from scipy import stats
                slope, _, r_value, _, _ = stats.linregress(range(len(serie.dropna())), serie.dropna())
                return {
                    'pendiente': slope,
                    'r_cuadrado': r_value**2
                }
            except:
                return {}
        return {}
    
    def _generar_recomendaciones(self, analisis: dict):
        """Genera recomendaciones de feature engineering basadas en el análisis"""
        recomendaciones = []
        
        # Recomendaciones basadas en tipos de datos
        if analisis['tipos_datos']['temporales']:
            recomendaciones.append({
                'tipo': 'temporal',
                'descripcion': 'Crear features temporales',
                'columnas': analisis['tipos_datos']['temporales']
            })
        
        # Recomendaciones para alta cardinalidad
        if analisis['tipos_datos']['alta_cardinalidad']:
            recomendaciones.append({
                'tipo': 'encoding',
                'descripcion': 'Usar target encoding para variables de alta cardinalidad',
                'columnas': analisis['tipos_datos']['alta_cardinalidad']
            })
        
        # Recomendaciones para outliers
        outliers_significativos = {k: v for k, v in analisis['calidad_datos']['outliers'].items() 
                                 if v['porcentaje'] > 0.05}
        if outliers_significativos:
            recomendaciones.append({
                'tipo': 'outliers',
                'descripcion': 'Tratar outliers significativos',
                'columnas': list(outliers_significativos.keys())
            })
        
        # Recomendaciones para correlaciones
        if analisis['calidad_datos']['correlaciones']:
            recomendaciones.append({
                'tipo': 'correlacion',
                'descripcion': 'Considerar reducción de dimensionalidad o selección de features',
                'pares': list(analisis['calidad_datos']['correlaciones'].keys())
            })
        
        # Recomendaciones para valores nulos
        nulos_significativos = {k: v for k, v in analisis['calidad_datos']['valores_nulos'].items() 
                              if v / analisis['estructura']['n_registros'] > 0.1}
        if nulos_significativos:
            recomendaciones.append({
                'tipo': 'valores_nulos',
                'descripcion': 'Implementar estrategias avanzadas de imputación',
                'columnas': list(nulos_significativos.keys())
            })
        
        analisis['recomendaciones'] = recomendaciones

class FeatureEngineering:
    """Clase para manejo de ingeniería de características"""
    
    def __init__(self):
        self.transformaciones = {}
        self.features_seleccionados = None
        self.pca = None
        self.outlier_detector = OutlierDetector()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.dataset_analyzer = DatasetAnalyzer()
        
    def procesar_dataset(self, df: pd.DataFrame, objetivo: str = None) -> tuple:
        """
        Procesa automáticamente el dataset aplicando las mejores estrategias
        
        Args:
            df: DataFrame a procesar
            objetivo: Variable objetivo (opcional)
            
        Returns:
            tuple: (DataFrame procesado, reporte de transformaciones)
        """
        try:
            # 1. Analizar dataset
            analisis = self.dataset_analyzer.analizar_dataset(df)
            
            # 2. Aplicar transformaciones recomendadas
            df_procesado = df.copy()
            transformaciones_aplicadas = []
            
            for recomendacion in analisis['recomendaciones']:
                if recomendacion['tipo'] == 'temporal':
                    for col in recomendacion['columnas']:
                        df_procesado = self.crear_features_temporales(df_procesado, col)
                        transformaciones_aplicadas.append(f"Features temporales creadas para {col}")
                
                elif recomendacion['tipo'] == 'encoding':
                    for col in recomendacion['columnas']:
                        df_procesado = self._aplicar_target_encoding(df_procesado, col, objetivo)
                        transformaciones_aplicadas.append(f"Target encoding aplicado a {col}")
                
                elif recomendacion['tipo'] == 'outliers':
                    df_procesado = self.outlier_detector.process(df_procesado)
                    transformaciones_aplicadas.append("Outliers procesados")
                
                elif recomendacion['tipo'] == 'correlacion':
                    if len(analisis['tipos_datos']['numericas']) > 10:
                        df_procesado = self.reducir_dimensionalidad(df_procesado)
                        transformaciones_aplicadas.append("PCA aplicado para reducir dimensionalidad")
            
            # 3. Crear features de interacción si hay numéricas
            if len(analisis['tipos_datos']['numericas']) > 1:
                df_procesado = self.crear_features_interaccion(
                    df_procesado, 
                    analisis['tipos_datos']['numericas']
                )
                transformaciones_aplicadas.append("Features de interacción creadas")
            
            # 4. Seleccionar mejores features si hay objetivo
            if objetivo and len(df_procesado.columns) > 10:
                df_procesado = self.seleccionar_features(df_procesado, objetivo)
                transformaciones_aplicadas.append("Selección de features realizada")
            
            # 5. Generar reporte
            reporte = self.generar_reporte_features(df, df_procesado)
            reporte['analisis_inicial'] = analisis
            reporte['transformaciones_aplicadas'] = transformaciones_aplicadas
            
            return df_procesado, reporte
            
        except Exception as e:
            logger.error(f"Error en procesamiento automático: {str(e)}")
            return df, {'error': str(e)}
    
    def _aplicar_target_encoding(self, df: pd.DataFrame, columna: str, objetivo: str) -> pd.DataFrame:
        """Aplica target encoding a una columna categórica"""
        if objetivo and objetivo in df.columns:
            try:
                media_global = df[objetivo].mean()
                encoding = df.groupby(columna)[objetivo].agg(['mean', 'count']).reset_index()
                # Aplicar suavizado
                smoothing = 100
                encoding['encoded'] = (encoding['count'] * encoding['mean'] + smoothing * media_global) / (encoding['count'] + smoothing)
                return df.merge(encoding[[columna, 'encoded']], on=columna, how='left')
            except:
                return df
        return df
    
    def crear_features_temporales(self, df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
        """Crea características basadas en fechas"""
        try:
            df_copy = df.copy()
            df_copy[columna_fecha] = pd.to_datetime(df_copy[columna_fecha])
            
            # Extraer componentes temporales
            df_copy[f'{columna_fecha}_año'] = df_copy[columna_fecha].dt.year
            df_copy[f'{columna_fecha}_mes'] = df_copy[columna_fecha].dt.month
            df_copy[f'{columna_fecha}_dia'] = df_copy[columna_fecha].dt.day
            df_copy[f'{columna_fecha}_dia_semana'] = df_copy[columna_fecha].dt.dayofweek
            
            return df_copy
        except Exception as e:
            logger.error(f"Error creando features temporales: {str(e)}")
            return df
    
    def crear_features_interaccion(self, df: pd.DataFrame, columnas: list) -> pd.DataFrame:
        """Crea características de interacción entre variables"""
        try:
            df_copy = df.copy()
            
            for i in range(len(columnas)):
                for j in range(i+1, len(columnas)):
                    col1, col2 = columnas[i], columnas[j]
                    if df_copy[col1].dtype in ['int64', 'float64'] and df_copy[col2].dtype in ['int64', 'float64']:
                        nombre = f'interaccion_{col1}_{col2}'
                        df_copy[nombre] = df_copy[col1] * df_copy[col2]
            
            return df_copy
        except Exception as e:
            logger.error(f"Error creando features de interacción: {str(e)}")
            return df
    
    def crear_features_agregacion(self, df: pd.DataFrame, grupo_por: str, columnas: list) -> pd.DataFrame:
        """Crea características basadas en agregaciones"""
        try:
            df_copy = df.copy()
            
            for col in columnas:
                if df_copy[col].dtype in ['int64', 'float64']:
                    # Calcular agregaciones
                    aggs = df_copy.groupby(grupo_por)[col].agg(['mean', 'std', 'min', 'max']).reset_index()
                    
                    # Renombrar columnas
                    aggs.columns = [grupo_por] + [f'{col}_{agg}' for agg in ['mean', 'std', 'min', 'max']]
                    
                    # Unir con el dataframe original
                    df_copy = df_copy.merge(aggs, on=grupo_por, how='left')
            
            return df_copy
        except Exception as e:
            logger.error(f"Error creando features de agregación: {str(e)}")
            return df
    
    def seleccionar_features(self, datos, objetivo, k=5):
        """
        Selecciona las k mejores features usando correlación con el objetivo
        
        Args:
            datos: DataFrame con los datos
            objetivo: Variable objetivo
            k: Número de features a seleccionar
            
        Returns:
            DataFrame con las k mejores features
        """
        try:
            # Separar features y objetivo
            X = datos.drop(columns=[objetivo])
            y = datos[objetivo]
            
            # Preparar datos para selección
            X_prep = X.copy()
            
            # Procesar columnas numéricas y categóricas por separado
            for col in X.columns:
                if X[col].dtype in ['object', 'category']:
                    # Para categóricas, usar la moda
                    X_prep[col] = X[col].fillna(X[col].mode()[0])
                    # Convertir a numérico usando LabelEncoder
                    le = LabelEncoder()
                    X_prep[col] = le.fit_transform(X_prep[col].astype(str))
                else:
                    # Para numéricas, usar la media
                    X_prep[col] = X[col].fillna(X[col].mean())
            
            # Preparar selector según tipo de objetivo
            if y.dtype == 'object' or len(y.unique()) < 10:
                selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
            
            # Seleccionar features
            selector.fit(X_prep, y)
            
            # Obtener nombres de features seleccionadas
            mask = selector.get_support()
            features_seleccionadas = X.columns[mask].tolist()
            
            # Retornar dataset con features seleccionadas y objetivo
            return datos[features_seleccionadas + [objetivo]]
            
        except Exception as e:
            logger.error(f"Error en selección de features: {str(e)}")
            return datos
    
    def reducir_dimensionalidad(self, df: pd.DataFrame, n_componentes: int = None, varianza_explicada: float = 0.95) -> pd.DataFrame:
        """
        Reduce dimensionalidad usando PCA
        
        Args:
            df: DataFrame con los datos
            n_componentes: Número de componentes (si es None, usa varianza_explicada)
            varianza_explicada: Varianza explicada deseada (entre 0 y 1)
            
        Returns:
            DataFrame con dimensionalidad reducida
        """
        try:
            # Seleccionar solo columnas numéricas
            numericas = df.select_dtypes(include=['int64', 'float64'])
            if len(numericas.columns) == 0:
                return df
            
            # Escalar datos
            X_scaled = self.scaler.fit_transform(numericas)
            
            # Configurar PCA
            if n_componentes is None:
                pca = PCA(n_components=varianza_explicada, svd_solver='full')
            else:
                pca = PCA(n_components=n_componentes)
            
            # Aplicar PCA
            X_pca = pca.fit_transform(X_scaled)
            
            # Crear DataFrame con componentes principales
            cols_pca = [f"PC{i+1}" for i in range(X_pca.shape[1])]
            df_pca = pd.DataFrame(X_pca, columns=cols_pca, index=df.index)
            
            # Combinar con variables no numéricas
            categoricas = df.select_dtypes(exclude=['int64', 'float64'])
            if not categoricas.empty:
                df_pca = pd.concat([df_pca, categoricas], axis=1)
            
            return df_pca
            
        except Exception as e:
            logger.error(f"Error en reducción de dimensionalidad: {str(e)}")
            return df
    
    def generar_reporte_features(self, df_original: pd.DataFrame, df_final: pd.DataFrame) -> dict:
        """
        Genera un reporte del procesamiento de features
        
        Args:
            df_original: DataFrame original
            df_final: DataFrame después del feature engineering
            
        Returns:
            dict: Reporte con métricas y cambios realizados
        """
        try:
            reporte = {
                'features_originales': {
                    'numero': len(df_original.columns),
                    'tipos': df_original.dtypes.value_counts().to_dict(),
                    'lista': df_original.columns.tolist()
                },
                'features_finales': {
                    'numero': len(df_final.columns),
                    'tipos': df_final.dtypes.value_counts().to_dict(),
                    'lista': df_final.columns.tolist()
                },
                'transformaciones': {
                    'features_seleccionados': [col for col in df_final.columns 
                                             if col in df_original.columns],
                    'pca_varianza_explicada': None  # Se actualiza si se usa PCA
                }
            }
            
            return reporte
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            return {} 