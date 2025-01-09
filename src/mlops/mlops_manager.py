import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import pandas as pd
from ..core.modelo_base import ModeloBase

logger = logging.getLogger(__name__)

class MLOpsManager:
    """Gestor de MLOps para modelos"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el gestor de MLOps
        
        Args:
            config: Configuración del gestor
        """
        self.config = config or {}
        self.ruta_base = os.path.abspath(os.path.join(os.getcwd(), 'mlops'))
        self.ruta_modelos = os.path.join(self.ruta_base, 'modelos')
        self.ruta_metricas = os.path.join(self.ruta_base, 'metricas')
        self.ruta_logs = os.path.join(self.ruta_base, 'logs')
        
        # Crear directorios si no existen
        for ruta in [self.ruta_modelos, self.ruta_metricas, self.ruta_logs]:
            os.makedirs(ruta, exist_ok=True)
    
    def registrar_modelo(self, modelo: ModeloBase, dataset: str, version: str) -> str:
        """
        Registra un modelo entrenado
        
        Args:
            modelo: Modelo a registrar
            dataset: Nombre del dataset usado
            version: Versión del modelo
            
        Returns:
            ID del modelo registrado
        """
        try:
            if not modelo.esta_entrenado:
                raise ValueError("El modelo debe estar entrenado antes de registrarlo")
            
            # Generar ID único
            modelo_id = self._generar_modelo_id(modelo.nombre, dataset, version)
            ruta_modelo = os.path.join(self.ruta_modelos, modelo_id)
            
            # Guardar modelo
            modelo.guardar(ruta_modelo)
            
            # Registrar métricas
            self._registrar_metricas(modelo_id, modelo.metricas)
            
            logger.info(f"Modelo registrado con ID: {modelo_id}")
            return modelo_id
            
        except Exception as e:
            logger.error(f"Error registrando modelo: {str(e)}")
            raise
    
    def cargar_modelo(self, modelo_id: str) -> ModeloBase:
        """
        Carga un modelo registrado
        
        Args:
            modelo_id: ID del modelo a cargar
            
        Returns:
            Modelo cargado
        """
        try:
            ruta_modelo = os.path.join(self.ruta_modelos, modelo_id)
            
            if not os.path.exists(ruta_modelo):
                raise FileNotFoundError(f"No se encontró el modelo con ID: {modelo_id}")
            
            # Cargar metadata
            with open(os.path.join(ruta_modelo, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Instanciar modelo del tipo correcto
            modelo_clase = self._obtener_clase_modelo(metadata['tipo'])
            modelo = modelo_clase(nombre=metadata['nombre'], tipo=metadata['tipo'], config=metadata['config'])
            
            # Cargar modelo
            modelo.cargar(ruta_modelo)
            
            logger.info(f"Modelo cargado: {modelo_id}")
            return modelo
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def obtener_metricas(self, modelo_id: str) -> Dict[str, Any]:
        """
        Obtiene las métricas de un modelo
        
        Args:
            modelo_id: ID del modelo
            
        Returns:
            Diccionario con métricas
        """
        try:
            ruta_metricas = os.path.join(self.ruta_metricas, f"metricas_{modelo_id}.json")
            
            if not os.path.exists(ruta_metricas):
                raise FileNotFoundError(f"No se encontraron métricas para el modelo: {modelo_id}")
            
            with open(ruta_metricas, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {str(e)}")
            raise
    
    def listar_modelos(self) -> List[Dict[str, Any]]:
        """
        Lista todos los modelos registrados
        
        Returns:
            Lista de metadatos de modelos
        """
        try:
            modelos = []
            
            for modelo_id in os.listdir(self.ruta_modelos):
                ruta_metadata = os.path.join(self.ruta_modelos, modelo_id, 'metadata.json')
                
                if os.path.exists(ruta_metadata):
                    with open(ruta_metadata, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        metadata['id'] = modelo_id
                        modelos.append(metadata)
            
            return modelos
            
        except Exception as e:
            logger.error(f"Error listando modelos: {str(e)}")
            raise
    
    def _generar_modelo_id(self, nombre: str, dataset: str, version: str) -> str:
        """
        Genera un ID único para el modelo
        
        Args:
            nombre: Nombre del modelo
            dataset: Nombre del dataset
            version: Versión del modelo
            
        Returns:
            ID único del modelo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"modelo_{nombre}_{dataset}_v{version}_{timestamp}"
    
    def _registrar_metricas(self, modelo_id: str, metricas: Dict[str, Any]) -> None:
        """
        Registra las métricas de un modelo
        
        Args:
            modelo_id: ID del modelo
            metricas: Métricas a registrar
        """
        try:
            ruta_metricas = os.path.join(self.ruta_metricas, f"metricas_{modelo_id}.json")
            
            with open(ruta_metricas, 'w', encoding='utf-8') as f:
                json.dump(metricas, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error registrando métricas: {str(e)}")
            raise
    
    def _obtener_clase_modelo(self, tipo: str) -> Any:
        """
        Obtiene la clase del modelo según su tipo
        
        Args:
            tipo: Tipo de modelo
            
        Returns:
            Clase del modelo
        """
        # Por ahora todos los modelos son H2O
        from ..modelos.h2o_modelo import ModeloH2O
        if tipo in ['clasificacion', 'regresion']:
            return ModeloH2O
        else:
            raise ValueError(f"Tipo de modelo no soportado: {tipo}")
            
    def generar_reporte_modelo(self, modelo_id: str) -> Dict[str, Any]:
        """
        Genera un reporte completo del modelo
        
        Args:
            modelo_id: ID del modelo
            
        Returns:
            Diccionario con el reporte
        """
        try:
            # Obtener metadata del modelo
            ruta_metadata = os.path.join(self.ruta_modelos, modelo_id, 'metadata.json')
            with open(ruta_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Obtener métricas
            metricas = self.obtener_metricas(modelo_id)
            
            # Generar reporte
            reporte = {
                'modelo_id': modelo_id,
                'metadata': metadata,
                'metricas': metricas,
                'fecha_reporte': datetime.now().isoformat()
            }
            
            # Guardar reporte
            ruta_reporte = os.path.join(self.ruta_base, 'reportes', f"reporte_{modelo_id}.json")
            os.makedirs(os.path.dirname(ruta_reporte), exist_ok=True)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                json.dump(reporte, f, indent=4, ensure_ascii=False)
            
            return reporte
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            raise 