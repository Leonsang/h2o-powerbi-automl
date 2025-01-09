import os
import json
import h2o
import pandas as pd
from datetime import datetime
from src.logger import Logger

logger = Logger('modelo_manager')

class ModeloManager:
    """Gestiona el ciclo de vida completo de los modelos"""
    
    def __init__(self, base_dir='./output'):
        self.logger = logger
        self.base_dir = base_dir
        self._crear_estructura_directorios()

    def _crear_estructura_directorios(self):
        """Crea la estructura de directorios necesaria"""
        try:
            # Estructura base
            directorios = [
                'modelos',
                'metricas',
                'interpretabilidad',
                'visualizaciones',
                'datos_procesados',
                'reportes'
            ]
            
            for dir_name in directorios:
                os.makedirs(os.path.join(self.base_dir, dir_name), exist_ok=True)
                
        except Exception as e:
            self.logger.error(f"Error creando estructura de directorios: {str(e)}")
            raise

    def crear_ejercicio_automl(self, nombre_ejercicio, descripcion=None):
        """
        Crea un nuevo ejercicio de AutoML
        
        Args:
            nombre_ejercicio: Identificador único del ejercicio
            descripcion: Descripción del ejercicio
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            ejercicio_id = f"{nombre_ejercicio}_{timestamp}"
            
            # Crear estructura para el ejercicio
            ejercicio_dir = os.path.join(self.base_dir, 'ejercicios', ejercicio_id)
            subdirs = [
                'modelos',
                'metricas',
                'interpretabilidad',
                'visualizaciones',
                'datos_procesados',
                'reportes'
            ]
            
            for subdir in subdirs:
                os.makedirs(os.path.join(ejercicio_dir, subdir), exist_ok=True)
            
            # Guardar metadata del ejercicio
            metadata = {
                'id': ejercicio_id,
                'nombre': nombre_ejercicio,
                'descripcion': descripcion,
                'fecha_creacion': timestamp,
                'estado': 'creado',
                'modelos': [],
                'metricas': {},
                'configuracion': {}
            }
            
            self._guardar_metadata(ejercicio_id, metadata)
            return ejercicio_id
            
        except Exception as e:
            self.logger.error(f"Error creando ejercicio AutoML: {str(e)}")
            raise

    def guardar_modelo(self, modelo, ejercicio_id, nombre, metricas=None):
        """
        Guarda un modelo y sus metadatos asociados
        
        Args:
            modelo: Modelo H2O entrenado
            ejercicio_id: ID del ejercicio
            nombre: Nombre del modelo
            metricas: Diccionario de métricas
        """
        try:
            # Rutas
            modelo_dir = os.path.join(self.base_dir, 'ejercicios', ejercicio_id, 'modelos')
            ruta_modelo = os.path.join(modelo_dir, nombre)
            
            # Guardar modelo
            h2o.save_model(modelo, ruta_modelo)
            
            # Actualizar metadata
            metadata = self._cargar_metadata(ejercicio_id)
            metadata['modelos'].append({
                'nombre': nombre,
                'ruta': ruta_modelo,
                'tipo': modelo.__class__.__name__,
                'fecha_guardado': datetime.now().strftime("%Y%m%d_%H%M"),
                'metricas': metricas
            })
            
            self._guardar_metadata(ejercicio_id, metadata)
            
        except Exception as e:
            self.logger.error(f"Error guardando modelo: {str(e)}")
            raise

    def guardar_resultados(self, ejercicio_id, tipo, resultados):
        """
        Guarda resultados de análisis
        
        Args:
            ejercicio_id: ID del ejercicio
            tipo: Tipo de resultados ('metricas', 'interpretabilidad', etc.)
            resultados: Diccionario con resultados
        """
        try:
            ruta = os.path.join(self.base_dir, 'ejercicios', ejercicio_id, tipo)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            nombre_archivo = f"{tipo}_{timestamp}.json"
            
            with open(os.path.join(ruta, nombre_archivo), 'w') as f:
                json.dump(resultados, f, indent=4)
                
            # Actualizar metadata
            metadata = self._cargar_metadata(ejercicio_id)
            if tipo not in metadata:
                metadata[tipo] = []
            metadata[tipo].append(nombre_archivo)
            
            self._guardar_metadata(ejercicio_id, metadata)
            
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {str(e)}")
            raise

    def cargar_modelo(self, ejercicio_id, nombre_modelo):
        """Carga un modelo específico"""
        try:
            metadata = self._cargar_metadata(ejercicio_id)
            modelo_info = next(
                (m for m in metadata['modelos'] if m['nombre'] == nombre_modelo),
                None
            )
            
            if not modelo_info:
                raise ValueError(f"Modelo {nombre_modelo} no encontrado")
                
            return h2o.load_model(modelo_info['ruta'])
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {str(e)}")
            raise

    def obtener_resultados(self, ejercicio_id, tipo):
        """Obtiene resultados de un tipo específico"""
        try:
            ruta = os.path.join(self.base_dir, 'ejercicios', ejercicio_id, tipo)
            resultados = []
            
            for archivo in os.listdir(ruta):
                if archivo.endswith('.json'):
                    with open(os.path.join(ruta, archivo), 'r') as f:
                        resultados.append(json.load(f))
                        
            return resultados
            
        except Exception as e:
            self.logger.error(f"Error obteniendo resultados: {str(e)}")
            raise

    def listar_ejercicios(self):
        """Lista todos los ejercicios de AutoML"""
        try:
            ejercicios_dir = os.path.join(self.base_dir, 'ejercicios')
            ejercicios = []
            
            for ejercicio_id in os.listdir(ejercicios_dir):
                metadata = self._cargar_metadata(ejercicio_id)
                ejercicios.append(metadata)
                
            return ejercicios
            
        except Exception as e:
            self.logger.error(f"Error listando ejercicios: {str(e)}")
            raise

    def _guardar_metadata(self, ejercicio_id, metadata):
        """Guarda metadata de un ejercicio"""
        try:
            ruta = os.path.join(self.base_dir, 'ejercicios', ejercicio_id, 'metadata.json')
            with open(ruta, 'w') as f:
                json.dump(metadata, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error guardando metadata: {str(e)}")
            raise

    def _cargar_metadata(self, ejercicio_id):
        """Carga metadata de un ejercicio"""
        try:
            ruta = os.path.join(self.base_dir, 'ejercicios', ejercicio_id, 'metadata.json')
            with open(ruta, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error cargando metadata: {str(e)}")
            raise 