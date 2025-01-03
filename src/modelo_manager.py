import os
import h2o
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from .logger import Logger

class ModeloManager:
    def __init__(self):
        self.logger = Logger('modelo_manager')
        self.tipos_modelo = [
            'automl', 'gbm', 'rf', 'glm', 'deeplearning', 
            'xgboost', 'lightgbm', 'stackedensemble', 'drf'
        ]
        self.modelos_activos = {}
        self.historial = []

    def crear_estructura_directorios(self, ruta_base):
        """Crea la estructura de directorios para el modelo"""
        directorios = ['modelo', 'metricas', 'graficos', 'resultados']
        for dir in directorios:
            Path(os.path.join(ruta_base, dir)).mkdir(parents=True, exist_ok=True)
        return {dir: os.path.join(ruta_base, dir) for dir in directorios}

    def guardar_modelo(self, modelo, tipo_modelo, metricas=None):
        """Guarda el modelo y todos sus artefactos asociados"""
        if tipo_modelo not in self.tipos_modelo:
            raise ValueError(f"Tipo de modelo no válido: {tipo_modelo}")
            
        # 1. Crear estructura de directorios
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_base = f"modelos/{tipo_modelo}_{timestamp}"
        dirs = self.crear_estructura_directorios(ruta_base)
        
        try:
            self.logger.info(f"Guardando modelo tipo: {tipo_modelo}")
            # 2. Guardar modelo H2O
            h2o.save_model(modelo, path=dirs['modelo'])
            
            # 3. Guardar métricas
            if metricas:
                # Métricas básicas
                with open(os.path.join(dirs['metricas'], 'basic.json'), 'w') as f:
                    json.dump(metricas.get('metricas_basicas', {}), f, indent=4)
                
                # Métricas H2O
                with open(os.path.join(dirs['metricas'], 'h2o_metrics.json'), 'w') as f:
                    json.dump(metricas.get('h2o_metrics', {}), f, indent=4)
                
                # Cross-validation
                with open(os.path.join(dirs['metricas'], 'cross_val.json'), 'w') as f:
                    json.dump(metricas.get('cross_validation', {}), f, indent=4)
            
            # 4. Guardar resultados
            if 'leaderboard' in metricas:
                pd.DataFrame(metricas['leaderboard']).to_csv(
                    os.path.join(dirs['resultados'], 'leaderboard.csv')
                )
            
            # 5. Guardar metadata
            metadata = {
                'timestamp': timestamp,
                'tipo_modelo': tipo_modelo,
                'ruta_base': ruta_base,
                'model_type': metricas.get('model_type'),
                'training_time': metricas.get('training_time')
            }
            
            with open(os.path.join(dirs['modelo'], 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # 6. Actualizar historial
            self.historial.append(metadata)
            
            return ruta_base
            
        except Exception as e:
            self.logger.error(f"Error guardando modelo: {str(e)}")
            return None

    def cargar_modelo(self, ruta):
        """Carga un modelo guardado"""
        try:
            return h2o.load_model(ruta)
        except Exception as e:
            print(f"Error cargando modelo: {str(e)}")
            return None

    def registrar_modelo(self, modelo, tipo_modelo, nombre=None):
        """Registra un modelo para uso activo"""
        if nombre is None:
            nombre = f"{tipo_modelo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.modelos_activos[nombre] = modelo
        return nombre

    def analizar_modelo(self, modelo, datos, predicciones, tipo_modelo='automl'):
        """Analiza un modelo específico"""
        return analizar_resultados(modelo, datos, predicciones, tipo_modelo)

    def comparar_modelos_activos(self):
        """Compara todos los modelos activos"""
        return comparar_modelos(self.modelos_activos)

    def limpiar_modelos_antiguos(self, dias=30):
        """Limpia modelos más antiguos que el número de días especificado"""
        # Implementar limpieza de modelos antiguos
        pass 