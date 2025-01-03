from setuptools import setup, find_packages
import os
from pathlib import Path

def read_requirements():
    """Lee los requisitos desde requirements.txt"""
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def create_directories():
    """Crea la estructura inicial de directorios"""
    dirs = [
        'modelos',
        'logs/h2o',
        'logs/modelos',
        'logs/tests',
        'temp/h2o_temp',
        'datos'
    ]
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)

# Crear estructura de directorios
create_directories()

setup(
    name="h2o_powerbi",
    version="1.0.0",
    description="Integración H2O AutoML con Power BI",
    author="Erick Sang",
    author_email="tu.email@dominio.com",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=read_requirements(),
    
    # Metadatos adicionales
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Data Scientists',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    # Datos y archivos adicionales
    package_data={
        'h2o_powerbi': [
            'config/*.json',
            'datos/*.csv',
            'docs/*.md'
        ]
    },
    
    # Scripts de entrada
    entry_points={
        'console_scripts': [
            'h2o-pbi=src.script_pbi:main',
            'h2o-pbi-clean=config.limpiar_todo:main',
            'h2o-pbi-init=config.crear_directorios:main'
        ]
    },
    
    # URLs del proyecto
    project_urls={
        'Source': 'https://github.com/tu_usuario/h2o_powerbi',
        'Documentation': 'https://github.com/tu_usuario/h2o_powerbi/docs',
        'Bug Reports': 'https://github.com/tu_usuario/h2o_powerbi/issues'
    },
    
    # Dependencias extras para desarrollo
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-html>=4.1.0',
            'coverage>=7.3.0',
            'black>=23.0.0',
            'flake8>=6.0.0'
        ]
    }
) 