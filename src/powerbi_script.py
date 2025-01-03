import sys
import os
import pandas as pd
from src.IntegradorH2O_PBI import H2OModeloAvanzado

def main(dataset=None):
    """
    Integrador entre Power BI y el proyecto H2O
    """
    try:
        if dataset is None:
            return pd.DataFrame({'Error': ['No hay datos']})
            
        modelo = H2OModeloAvanzado()
        resultado = modelo.entrenar(datos=dataset)
        return resultado
        
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]}) 