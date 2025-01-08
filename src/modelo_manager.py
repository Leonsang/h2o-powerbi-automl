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

    def guardar_modelo(self, modelo, nombre, metricas):
        """Guarda el modelo con un nombre descriptivo"""
        try:
            # Crear directorio si no existe
            if not os.path.exists(self.directorio_modelos):
                os.makedirs(self.directorio_modelos)
            
            # Guardar modelo
            ruta_modelo = os.path.join(self.directorio_modelos, f"{nombre}.zip")
            h2o.save_model(modelo, ruta_modelo)
            
            # Guardar métricas
            ruta_metricas = os.path.join(self.directorio_modelos, f"{nombre}_metricas.json")
            with open(ruta_metricas, 'w') as f:
                json.dump(metricas, f)
                
            self.logger.info(f"Modelo guardado: {nombre}")
            
        except Exception as e:
            self.logger.error(f"Error al guardar modelo: {str(e)}")
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