from setuptools import setup, find_packages

setup(
    name="h2o_powerbi",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'h2o>=3.40.0',
        'pandas>=1.0.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0'
    ],
    author="Erick Sang",
    description="Integración H2O AutoML con Power BI",
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'h2o-pbi-init=config.crear_directorios:crear_estructura_proyecto',
            'h2o-pbi-clean=config.crear_directorios:limpiar_directorios_temp'
        ]
    }
) 