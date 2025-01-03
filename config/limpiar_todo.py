import os
import shutil
from pathlib import Path
from datetime import datetime
from src.logger import Logger

class LimpiadorProyecto:
    def __init__(self):
        self.logger = Logger('limpiador')
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Directorios y patrones a limpiar
        self.dirs_limpiar = {
            'temp': {
                'path': 'temp/h2o_temp',
                'patron': '*',
                'preservar_dir': True
            },
            'logs_antiguos': {
                'path': 'logs',
                'patron': '*.log',
                'dias_max': 30,
                'preservar_dir': True
            },
            'cache_python': {
                'paths': [
                    '__pycache__',
                    'src/__pycache__',
                    'tests/__pycache__',
                    'config/__pycache__'
                ],
                'preservar_dir': False
            },
            'build': {
                'paths': ['build', 'dist', '*.egg-info'],
                'preservar_dir': False
            }
        }

    def limpiar_directorio(self, path, patron='*', dias_max=None, preservar_dir=True):
        """Limpia un directorio según los criterios especificados"""
        try:
            path_completo = os.path.join(self.root_dir, path)
            if not os.path.exists(path_completo):
                return
                
            self.logger.info(f"Limpiando {path_completo}")
            
            if dias_max:
                # Eliminar archivos más antiguos que dias_max
                fecha_limite = datetime.now().timestamp() - (dias_max * 24 * 3600)
                for file in Path(path_completo).glob(patron):
                    if file.stat().st_mtime < fecha_limite:
                        file.unlink()
                        self.logger.info(f"Eliminado archivo antiguo: {file}")
            else:
                # Eliminar todo según el patrón
                for item in Path(path_completo).glob(patron):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir() and not preservar_dir:
                        shutil.rmtree(item)
                    self.logger.info(f"Eliminado: {item}")
                        
            # Eliminar directorio si está vacío y no se debe preservar
            if not preservar_dir and os.path.exists(path_completo) and not os.listdir(path_completo):
                shutil.rmtree(path_completo)
                self.logger.info(f"Eliminado directorio vacío: {path_completo}")
                
        except Exception as e:
            self.logger.error(f"Error limpiando {path}: {str(e)}")

    def limpiar_todo(self):
        """Ejecuta la limpieza completa del proyecto"""
        self.logger.info("Iniciando limpieza del proyecto")
        
        # Limpiar temp
        self.limpiar_directorio(**self.dirs_limpiar['temp'])
        
        # Limpiar logs antiguos
        self.limpiar_directorio(**self.dirs_limpiar['logs_antiguos'])
        
        # Limpiar cache Python
        for path in self.dirs_limpiar['cache_python']['paths']:
            self.limpiar_directorio(path, preservar_dir=False)
            
        # Limpiar archivos build
        for path in self.dirs_limpiar['build']['paths']:
            self.limpiar_directorio(path, preservar_dir=False)
            
        self.logger.info("Limpieza completada")

def main():
    # Pedir confirmación
    respuesta = input("⚠️  ¿Estás seguro de que quieres limpiar todo? (s/N): ")
    if respuesta.lower() == 's':
        limpiador = LimpiadorProyecto()
        limpiador.limpiar_todo()
    else:
        print("Operación cancelada")

if __name__ == "__main__":
    main() 