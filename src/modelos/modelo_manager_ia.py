import os
import requests
import hashlib
from tqdm import tqdm
from src.logger import Logger

logger = Logger('modelo_manager_ia')

class ModeloManagerIA:
    MODELOS_DISPONIBLES = {
        'gpt4all-j': {
            'url': 'https://gpt4all.io/models/ggml-gpt4all-j.bin',
            'md5': 'dc86e4b0c1c29c51d6f4c2eadf54c65c',
            'size': 3785248280  # ~3.8GB
        },
        'orca-mini': {
            'url': 'https://gpt4all.io/models/ggml-orca-mini-3b.bin',
            'md5': '42c4be474292a9d57f8432c8817430c1',
            'size': 1928589312  # ~1.9GB
        }
    }

    def __init__(self, modelo_base='orca-mini'):
        """Inicializa el gestor de modelos de IA"""
        self.modelo_base = modelo_base
        self.models_dir = os.path.join('models', 'ia')
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, f"{modelo_base}.bin")

    def verificar_modelo(self):
        """Verifica si el modelo está disponible y es válido"""
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"Modelo {self.modelo_base} no encontrado. Iniciando descarga...")
                self.descargar_modelo()
                return self.verificar_hash()
            
            if not self.verificar_hash():
                logger.warning("Hash del modelo no coincide. Redownloading...")
                self.descargar_modelo()
                return self.verificar_hash()
            
            logger.info(f"Modelo {self.modelo_base} verificado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error verificando modelo: {str(e)}")
            raise

    def descargar_modelo(self):
        """Descarga el modelo seleccionado"""
        try:
            info_modelo = self.MODELOS_DISPONIBLES[self.modelo_base]
            url = info_modelo['url']
            total_size = info_modelo['size']

            logger.info(f"Descargando modelo {self.modelo_base} ({total_size/1e9:.1f}GB)...")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            progress = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Descargando {self.modelo_base}"
            )

            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        progress.update(size)
            
            progress.close()
            logger.info("Descarga completada")
            
        except Exception as e:
            logger.error(f"Error descargando modelo: {str(e)}")
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            raise

    def verificar_hash(self):
        """Verifica el hash MD5 del modelo"""
        try:
            if not os.path.exists(self.model_path):
                return False
                
            hash_esperado = self.MODELOS_DISPONIBLES[self.modelo_base]['md5']
            
            md5_hash = hashlib.md5()
            with open(self.model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
                    
            return md5_hash.hexdigest() == hash_esperado
            
        except Exception as e:
            logger.error(f"Error verificando hash: {str(e)}")
            return False

    def get_model_path(self):
        """Retorna la ruta al modelo verificado"""
        if self.verificar_modelo():
            return self.model_path
        raise Exception("No se pudo verificar el modelo") 