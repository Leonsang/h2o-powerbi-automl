import os
import json
import h2o
import pandas as pd
from datetime import datetime
from .logger import Logger

class ModeloManager:
    """
    Gestiona el ciclo de vida completo de los modelos H2O.
    
    Responsabilidades:
    - Guardar y cargar modelos
    - Gestionar predicciones
    - Mantener métricas y metadata
    """
    
    def __init__(self):
        self.logger = Logger('modelo_manager')
        self.ruta_modelos = 'modelos'
        self.ruta_predicciones = os.path.join(self.ruta_modelos, 'predicciones')
        self.ultimo_modelo = None
        self.predicciones = {}
        self._crear_directorios()

    def _crear_directorios(self):
        """Asegura que existan los directorios necesarios"""
        os.makedirs(self.ruta_modelos, exist_ok=True)
        os.makedirs(self.ruta_predicciones, exist_ok=True)

    def guardar_modelo(self, modelo, tipo_modelo, metricas=None):
        """
        Guarda el modelo con su metadata asociada.
        
        Args:
            modelo: Modelo H2O entrenado
            tipo_modelo: Tipo de modelo (ej: 'automl', 'gbm')
            metricas: Dict con métricas del modelo
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            nombre_modelo = f"{tipo_modelo}_{timestamp}"
            
            # 1. Guardar modelo H2O
            ruta_modelo = h2o.save_model(
                model=modelo,
                path=self.ruta_modelos,
                force=True
            )
            
            # 2. Guardar metadata
            metadata = {
                'nombre': nombre_modelo,
                'tipo': tipo_modelo,
                'fecha': timestamp,
                'ruta': ruta_modelo,
                'metricas': metricas or {},
                'parametros': modelo.get_params()
            }
            
            ruta_metadata = os.path.join(
                self.ruta_modelos,
                f"{nombre_modelo}_metadata.json"
            )
            
            with open(ruta_metadata, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.ultimo_modelo = modelo
            self.logger.info(f"Modelo guardado: {nombre_modelo}")
            
            return ruta_modelo
            
        except Exception as e:
            self.logger.error(f"Error guardando modelo: {str(e)}")
            raise

    def obtener_prediccion(self, prediccion_id):
        """
        Obtiene una predicción específica.
        
        Args:
            prediccion_id: ID único de la predicción
            
        Returns:
            Dict con datos de la predicción o None si no existe
        """
        try:
            ruta_prediccion = os.path.join(
                self.ruta_predicciones,
                f"prediccion_{prediccion_id}.json"
            )
            
            if os.path.exists(ruta_prediccion):
                with open(ruta_prediccion, 'r') as f:
                    return json.load(f)
            
            return self.predicciones.get(prediccion_id)
            
        except Exception as e:
            self.logger.error(f"Error obteniendo predicción: {str(e)}")
            return None

    def obtener_ultimo_modelo(self):
        """
        Obtiene el último modelo entrenado.
        
        Returns:
            Modelo H2O o None si no hay modelo
        """
        try:
            if self.ultimo_modelo:
                return self.ultimo_modelo
                
            # Buscar último modelo guardado
            archivos = os.listdir(self.ruta_modelos)
            modelos = [f for f in archivos if f.endswith('.model')]
            
            if not modelos:
                return None
                
            ultimo_modelo = sorted(modelos)[-1]
            return h2o.load_model(os.path.join(self.ruta_modelos, ultimo_modelo))
            
        except Exception as e:
            self.logger.error(f"Error obteniendo último modelo: {str(e)}")
            return None

    def guardar_prediccion(self, prediccion_id, datos, resultado):
        """
        Guarda los resultados de una predicción.
        
        Args:
            prediccion_id: ID único
            datos: Datos usados
            resultado: Resultado de la predicción
        """
        try:
            prediccion = {
                'id': prediccion_id,
                'fecha': datetime.now().isoformat(),
                'datos': datos.to_dict() if isinstance(datos, pd.DataFrame) else datos,
                'resultado': resultado
            }
            
            ruta = os.path.join(
                self.ruta_predicciones,
                f"prediccion_{prediccion_id}.json"
            )
            
            with open(ruta, 'w') as f:
                json.dump(prediccion, f, indent=4)
                
            self.predicciones[prediccion_id] = prediccion
            self.logger.info(f"Predicción guardada: {prediccion_id}")
            
        except Exception as e:
            self.logger.error(f"Error guardando predicción: {str(e)}")
            raise 