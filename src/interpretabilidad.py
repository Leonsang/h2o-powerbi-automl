import shap
import lime
import lime.lime_tabular
import dice_ml
import numpy as np
from src.logger import Logger

logger = Logger('interpretabilidad')

class Interpretador:
    def calcular_shap_values(self, modelo, datos):
        """Calcula valores SHAP para interpretabilidad global"""
        try:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(datos)
            return {
                'values': shap_values,
                'explainer': explainer,
                'expected_value': explainer.expected_value
            }
        except Exception as e:
            logger.error(f"Error calculando SHAP values: {str(e)}")
            raise

    def generar_lime_explicaciones(self, modelo, datos, num_features=5):
        """Genera explicaciones LIME para casos individuales"""
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=datos.values,
                feature_names=datos.columns,
                mode='regression'
            )
            
            # Explicar algunos casos representativos
            explicaciones = []
            indices = np.random.choice(len(datos), 3)
            
            for idx in indices:
                exp = explainer.explain_instance(
                    datos.iloc[idx].values, 
                    modelo.predict,
                    num_features=num_features
                )
                explicaciones.append({
                    'caso': idx,
                    'explicacion': exp
                })
                
            return explicaciones
        except Exception as e:
            logger.error(f"Error generando explicaciones LIME: {str(e)}")
            raise

    def generar_counterfactuals(self, modelo, datos):
        """Genera ejemplos contrafactuales"""
        try:
            # Configurar DiCE
            dice_data = dice_ml.Data(
                dataframe=datos,
                continuous_features=datos.select_dtypes(include=[np.number]).columns.tolist(),
                outcome_name=modelo.target_name
            )
            
            dice_model = dice_ml.Model(model=modelo, backend='sklearn')
            explainer = dice_ml.Dice(dice_data, dice_model)
            
            # Generar counterfactuals para algunos casos
            counterfactuals = []
            indices = np.random.choice(len(datos), 3)
            
            for idx in indices:
                cf = explainer.generate_counterfactuals(
                    datos.iloc[idx:idx+1],
                    total_CFs=3,
                    desired_class='opposite'
                )
                counterfactuals.append({
                    'caso': idx,
                    'counterfactuals': cf
                })
                
            return counterfactuals
        except Exception as e:
            logger.error(f"Error generando counterfactuals: {str(e)}")
            raise 