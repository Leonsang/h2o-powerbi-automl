import os
import logging
import pandas as pd
from datetime import datetime
import h2o
import json
import requests
import hashlib
from tqdm import tqdm
from .modelo_manager_ia import ModeloManagerIA

logger = logging.getLogger(__name__)

class MLOpsManager:
    """Gestiona el ciclo de vida completo de modelos en producción"""
    
    TIPOS_MODELOS = {
        'h2o_automl': 'Modelo H2O AutoML',
        'h2o_gbm': 'Modelo H2O GBM',
        'h2o_glm': 'Modelo H2O GLM',
        'h2o_rf': 'Modelo H2O Random Forest',
        'h2o_xgboost': 'Modelo H2O XGBoost',
        'h2o_deeplearning': 'Modelo H2O Deep Learning',
        'sklearn_rf': 'Modelo Scikit-learn Random Forest',
        'sklearn_xgboost': 'Modelo Scikit-learn XGBoost',
        'sklearn_lightgbm': 'Modelo Scikit-learn LightGBM',
        'pytorch': 'Modelo PyTorch',
        'tensorflow': 'Modelo TensorFlow',
        'custom': 'Modelo Personalizado'
    }
    
    TIPOS_PROBLEMAS = {
        'clasificacion_binaria': 'Clasificación Binaria',
        'clasificacion_multiclase': 'Clasificación Multiclase',
        'regresion': 'Regresión',
        'series_temporales': 'Series Temporales',
        'clustering': 'Clustering',
        'nlp': 'Procesamiento de Lenguaje Natural',
        'vision': 'Visión por Computadora'
    }
    
    def __init__(self):
        """Inicializa el gestor MLOps"""
        self.base_dir = os.path.abspath(os.path.join(os.getcwd(), 'mlops'))
        self.modelos_dir = os.path.join(self.base_dir, 'modelos')
        self.metricas_dir = os.path.join(self.base_dir, 'metricas')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
        self.visualizaciones_dir = os.path.join(self.output_dir, 'visualizaciones')
        self.reportes_dir = os.path.join(self.output_dir, 'reportes')
        self.modelo_ia_manager = ModeloManagerIA()
        
        # Crear directorios si no existen
        for dir in [self.base_dir, self.modelos_dir, self.metricas_dir, self.logs_dir, 
                   self.output_dir, self.visualizaciones_dir, self.reportes_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
                logger.info(f"Directorio creado: {dir}")

    def _detectar_tipo_modelo(self, modelo):
        """Detecta automáticamente el tipo de modelo"""
        try:
            if hasattr(modelo, '_model_json'):
                # Es un modelo H2O
                algo = modelo._model_json['algo']
                if algo == 'gbm':
                    return 'h2o_gbm'
                elif algo == 'glm':
                    return 'h2o_glm'
                elif algo == 'drf':
                    return 'h2o_rf'
                elif algo == 'xgboost':
                    return 'h2o_xgboost'
                elif algo == 'deeplearning':
                    return 'h2o_deeplearning'
                return 'h2o_automl'
            
            # Otros tipos de modelos
            modelo_str = str(type(modelo)).lower()
            if 'sklearn' in modelo_str:
                if 'randomforest' in modelo_str:
                    return 'sklearn_rf'
                elif 'xgboost' in modelo_str:
                    return 'sklearn_xgboost'
                elif 'lightgbm' in modelo_str:
                    return 'sklearn_lightgbm'
            elif 'torch' in modelo_str:
                return 'pytorch'
            elif 'tensorflow' in modelo_str:
                return 'tensorflow'
            
            return 'custom'
        except:
            return 'custom'

    def _generar_modelo_id(self, tipo_modelo, tipo_problema, dataset, version='v1'):
        """Genera un ID único para el modelo con formato estandarizado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"modelo_{tipo_modelo}_{tipo_problema}_{dataset}_{version}_{timestamp}"

    def _generar_metricas_id(self, modelo_id):
        """Genera un ID único para las métricas"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"metricas_{modelo_id}_eval_{timestamp}"

    def registrar_modelo(self, modelo, metadata=None):
        """Registra un modelo en el sistema MLOps"""
        try:
            # Validar y obtener metadata
            if metadata is None:
                metadata = {}
            
            # Detectar tipo de modelo
            tipo_modelo = metadata.get('tipo_modelo', self._detectar_tipo_modelo(modelo))
            if tipo_modelo not in self.TIPOS_MODELOS:
                tipo_modelo = 'custom'
            
            # Validar tipo de problema
            tipo_problema = metadata.get('tipo_problema', 'clasificacion_binaria')
            if tipo_problema not in self.TIPOS_PROBLEMAS:
                tipo_problema = 'clasificacion_binaria'
            
            # Obtener información del dataset
            dataset = metadata.get('dataset', 'titanic')
            version = metadata.get('version', 'v1')
            
            # Generar ID único
            modelo_id = self._generar_modelo_id(tipo_modelo, tipo_problema, dataset, version)
            
            # Crear directorio para el modelo
            ruta_modelo = os.path.join(self.modelos_dir, modelo_id, 'modelo')
            os.makedirs(ruta_modelo, exist_ok=True)
            
            # Guardar modelo según su tipo
            if hasattr(modelo, 'save_model'):
                modelo_path = h2o.save_model(modelo, path=ruta_modelo, force=True)
            else:
                raise ValueError(f"Tipo de modelo no soportado: {tipo_modelo}")
            
            # Preparar metadata completa
            metadata_completa = {
                'modelo_id': modelo_id,
                'tipo_modelo': {
                    'id': tipo_modelo,
                    'nombre': self.TIPOS_MODELOS[tipo_modelo]
                },
                'tipo_problema': {
                    'id': tipo_problema,
                    'nombre': self.TIPOS_PROBLEMAS[tipo_problema]
                },
                'dataset': dataset,
                'version': version,
                'fecha_creacion': datetime.now().isoformat(),
                'ruta_modelo': modelo_path,
                'framework': 'h2o' if 'h2o_' in tipo_modelo else tipo_modelo.split('_')[0],
                'parametros_modelo': modelo._model_json['parameters'] if hasattr(modelo, '_model_json') else {},
                'metadata_adicional': metadata
            }
            
            # Guardar metadata
            metadata_path = os.path.join(self.modelos_dir, modelo_id, f"{modelo_id}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_completa, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Modelo {metadata_completa['tipo_modelo']['nombre']} registrado con ID: {modelo_id}")
            return modelo_id
            
        except Exception as e:
            logger.error(f"Error registrando modelo: {str(e)}")
            raise

    def guardar_metricas(self, modelo_id, metricas):
        """Guarda las métricas de un modelo"""
        try:
            # Generar ID único para las métricas
            metricas_id = self._generar_metricas_id(modelo_id)
            
            # Crear directorio si no existe
            os.makedirs(self.metricas_dir, exist_ok=True)
            
            # Preparar datos de métricas
            datos_metricas = {
                'modelo_id': modelo_id,
                'metricas_id': metricas_id,
                'fecha_evaluacion': datetime.now().isoformat(),
                'metricas': metricas
            }
            
            # Guardar métricas
            ruta_metricas = os.path.join(self.metricas_dir, f"{metricas_id}.json")
            with open(ruta_metricas, 'w', encoding='utf-8') as f:
                json.dump(datos_metricas, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Métricas guardadas: {ruta_metricas}")
            return metricas_id
            
        except Exception as e:
            self.logger.error(f"Error guardando métricas: {str(e)}")
            raise

    def cargar_modelo(self, modelo_id):
        """Carga un modelo registrado"""
        try:
            # Obtener directorio del modelo
            modelo_dir = os.path.join(self.modelos_dir, modelo_id)
            if not os.path.exists(modelo_dir):
                raise FileNotFoundError(f"Directorio del modelo no encontrado: {modelo_dir}")
            
            # Cargar metadata
            ruta_metadata = os.path.join(modelo_dir, "metadata.json")
            with open(ruta_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            tipo_modelo = metadata.get('tipo', 'h2o')
            ruta_modelo = metadata['ruta_modelo']
            
            if tipo_modelo == 'h2o':
                try:
                    # Verificar que el archivo del modelo existe
                    if not os.path.exists(ruta_modelo):
                        raise FileNotFoundError(f"Archivo del modelo no encontrado: {ruta_modelo}")
                    
                    # Cargar el modelo H2O
                    modelo = h2o.load_model(ruta_modelo)
                    logger.info(f"Modelo H2O cargado desde: {ruta_modelo}")
                    
                except Exception as e:
                    logger.error(f"Error al cargar modelo H2O: {str(e)}")
                    raise
            
            elif tipo_modelo == 'ia':
                # Verificar y cargar modelo de IA
                if self.modelo_ia_manager.verificar_modelo():
                    modelo = ruta_modelo
                else:
                    raise Exception("No se pudo verificar el modelo de IA")
            
            logger.info(f"Modelo {modelo_id} cargado exitosamente")
            return modelo
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise

    def monitorear_modelo(self, modelo_id, datos, objetivo=None):
        """Configura monitoreo para un modelo"""
        try:
            # Cargar metadata del modelo
            ruta_metadata = os.path.join(self.modelos_dir, modelo_id, "metadata.json")
            with open(ruta_metadata, 'r') as f:
                metadata = json.load(f)
            
            tipo_modelo = metadata.get('tipo', 'h2o')
            
            # Cargar modelo
            modelo = self.cargar_modelo(modelo_id)
            
            # Métricas comunes
            metricas = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'modelo_id': modelo_id,
                'tipo_modelo': tipo_modelo,
                'n_registros': len(datos)
            }
            
            if tipo_modelo == 'h2o':
                # Convertir a H2O Frame y obtener predicciones
                h2o_frame = h2o.H2OFrame(datos)
                predicciones = modelo.predict(h2o_frame)
                metricas['distribucion_predicciones'] = predicciones.as_data_frame().describe().to_dict()
            
            elif tipo_modelo == 'ia':
                # Métricas específicas para modelos de IA
                metricas['estado_modelo'] = 'verificado' if self.modelo_ia_manager.verificar_modelo() else 'no_verificado'
                metricas['hash_modelo'] = self.modelo_ia_manager.verificar_hash()
            
            # Guardar métricas
            ruta_metricas = os.path.join(
                self.metricas_dir,
                f"metricas_{modelo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(ruta_metricas, 'w') as f:
                json.dump(metricas, f, indent=4)
            
            logger.info(f"Monitoreo configurado para modelo {modelo_id}")
            
        except Exception as e:
            logger.error(f"Error monitoreando modelo: {str(e)}")
            raise

    def obtener_metricas_modelo(self, modelo_id):
        """Obtiene métricas históricas de un modelo"""
        try:
            metricas = []
            patron = f"metricas_{modelo_id}_*.json"
            
            for archivo in os.listdir(self.metricas_dir):
                if archivo.startswith(f"metricas_{modelo_id}"):
                    ruta = os.path.join(self.metricas_dir, archivo)
                    metrica = pd.read_json(ruta, orient='records')
                    metricas.append(metrica)
            
            if metricas:
                return pd.concat(metricas)
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {str(e)}")
            return pd.DataFrame()
    
    def detectar_drift(self, modelo_id, datos_nuevos):
        """Detecta drift en los datos"""
        try:
            # Cargar metadata del modelo
            ruta_metadata = os.path.join(self.modelos_dir, modelo_id, "metadata.json")
            metadata = pd.read_json(ruta_metadata, orient='records').iloc[0]
            
            # Obtener distribución original
            distribucion_original = metadata['distribucion_features']
            
            # Calcular distribución actual
            distribucion_actual = datos_nuevos.describe().to_dict()
            
            # Calcular diferencias
            diferencias = {}
            for feature in distribucion_original.keys():
                if feature in distribucion_actual:
                    diff_mean = abs(
                        distribucion_original[feature]['mean'] - 
                        distribucion_actual[feature]['mean']
                    )
                    diff_std = abs(
                        distribucion_original[feature]['std'] - 
                        distribucion_actual[feature]['std']
                    )
                    diferencias[feature] = {
                        'diff_mean': diff_mean,
                        'diff_std': diff_std
                    }
            
            return diferencias
            
        except Exception as e:
            logger.error(f"Error detectando drift: {str(e)}")
            return {} 