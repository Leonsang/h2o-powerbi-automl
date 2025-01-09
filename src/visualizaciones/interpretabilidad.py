import shap
import lime
import lime.lime_tabular
import dice_ml
import numpy as np
import pandas as pd
from src.logger import Logger

logger = Logger('interpretabilidad')

class InterpretabilidadManager:
    """Gestiona todas las funcionalidades de interpretabilidad del modelo"""
    
    CAPACIDADES_MODELOS = {
        'XGBRegressor': ['shap', 'feature_importance', 'pdp', 'lime'],
        'XGBClassifier': ['shap', 'feature_importance', 'pdp', 'lime'],
        'RandomForestRegressor': ['shap', 'feature_importance', 'pdp', 'lime'],
        'RandomForestClassifier': ['shap', 'feature_importance', 'pdp', 'lime'],
        'LGBMRegressor': ['shap', 'feature_importance', 'pdp', 'lime'],
        'LGBMClassifier': ['shap', 'feature_importance', 'pdp', 'lime'],
        'CatBoostRegressor': ['shap', 'feature_importance', 'pdp'],
        'CatBoostClassifier': ['shap', 'feature_importance', 'pdp'],
        'LinearRegression': ['coef', 'lime'],
        'LogisticRegression': ['coef', 'lime'],
        'Ridge': ['coef', 'lime'],
        'Lasso': ['coef', 'lime'],
        'ElasticNet': ['coef', 'lime'],
        'SVR': ['lime'],
        'SVC': ['lime'],
        'MLPRegressor': ['lime'],
        'MLPClassifier': ['lime'],
        'H2OEstimator': ['varimp', 'pdp', 'shap', 'lime']
    }
    
    def __init__(self):
        self.logger = logger

    def detectar_capacidades_modelo(self, modelo):
        """Detecta las capacidades de interpretabilidad del modelo"""
        try:
            nombre_modelo = modelo.__class__.__name__
            capacidades = set()

            # Capacidades predefinidas por tipo de modelo
            if nombre_modelo in self.CAPACIDADES_MODELOS:
                capacidades.update(self.CAPACIDADES_MODELOS[nombre_modelo])

            # Detección dinámica de capacidades
            if hasattr(modelo, 'feature_importances_'):
                capacidades.add('feature_importance')
            if hasattr(modelo, 'coef_'):
                capacidades.add('coef')
            if hasattr(modelo, 'predict_proba'):
                capacidades.add('probabilidad')
            if hasattr(modelo, 'varimp'):
                capacidades.add('varimp')

            return list(capacidades)
        except Exception as e:
            self.logger.error(f"Error detectando capacidades: {str(e)}")
            return []

    def analisis_global(self, modelo, datos, objetivo):
        """
        Realiza análisis global de interpretabilidad adaptado al tipo de modelo
        """
        try:
            capacidades = self.detectar_capacidades_modelo(modelo)
            resultados = {
                'capacidades_modelo': capacidades,
                'importancia_variables': None,
                'shap_values': None,
                'pdp': None,
                'coef': None
            }

            # Importancia de variables según el tipo de modelo
            if 'feature_importance' in capacidades:
                resultados['importancia_variables'] = self.calcular_importancia_variables(modelo, datos)
            elif 'varimp' in capacidades:
                resultados['importancia_variables'] = self._calcular_importancia_h2o(modelo)
            elif 'coef' in capacidades:
                resultados['coef'] = self._calcular_coeficientes(modelo, datos.columns)

            # SHAP values si está disponible
            if 'shap' in capacidades:
                try:
                    resultados['shap_values'] = self.calcular_shap_values(modelo, datos)
                except Exception as e:
                    self.logger.warning(f"Error calculando SHAP values: {str(e)}")

            # PDP si está disponible
            if 'pdp' in capacidades:
                try:
                    resultados['pdp'] = self.calcular_pdp(modelo, datos, objetivo)
                except Exception as e:
                    self.logger.warning(f"Error calculando PDP: {str(e)}")

            return resultados
        except Exception as e:
            self.logger.error(f"Error en análisis global: {str(e)}")
            raise

    def analisis_local(self, modelo, datos, indices=None, num_ejemplos=3):
        """
        Realiza análisis local adaptado al tipo de modelo
        """
        try:
            if indices is None:
                indices = np.random.choice(len(datos), num_ejemplos)
            
            capacidades = self.detectar_capacidades_modelo(modelo)
            resultados = {
                'capacidades_modelo': capacidades,
                'lime': None,
                'shap_local': None,
                'counterfactuals': None
            }
            
            # LIME siempre disponible
            resultados['lime'] = self.generar_lime_explicaciones(modelo, datos, indices)
            
            # SHAP local si está disponible
            if 'shap' in capacidades:
                try:
                    resultados['shap_local'] = self.calcular_shap_local(modelo, datos, indices)
                except Exception as e:
                    self.logger.warning(f"Error calculando SHAP local: {str(e)}")
            
            # Counterfactuals solo para clasificación
            if 'probabilidad' in capacidades:
                try:
                    resultados['counterfactuals'] = self.generar_counterfactuals(modelo, datos, indices)
                except Exception as e:
                    self.logger.warning(f"Error generando counterfactuals: {str(e)}")
            
            return resultados
        except Exception as e:
            self.logger.error(f"Error en análisis local: {str(e)}")
            raise

    def _calcular_importancia_h2o(self, modelo):
        """Calcula importancia de variables para modelos H2O"""
        try:
            return modelo.varimp(use_pandas=True)
        except Exception as e:
            self.logger.error(f"Error calculando importancia H2O: {str(e)}")
            raise

    def _calcular_coeficientes(self, modelo, columnas):
        """Calcula e interpreta coeficientes para modelos lineales"""
        try:
            coef = modelo.coef_ if len(modelo.coef_.shape) == 1 else modelo.coef_[0]
            return pd.DataFrame({
                'feature': columnas,
                'coefficient': coef,
                'abs_impact': np.abs(coef)
            }).sort_values('abs_impact', ascending=False)
        except Exception as e:
            self.logger.error(f"Error calculando coeficientes: {str(e)}")
            raise

    def calcular_shap_values(self, modelo, datos):
        """Calcula valores SHAP adaptado al tipo de modelo"""
        try:
            # Seleccionar el explainer adecuado según el tipo de modelo
            if hasattr(modelo, '_model_json'):  # H2O
                return self._calcular_shap_h2o(modelo, datos)
            elif hasattr(modelo, 'predict_proba'):  # Tree-based classifier
                explainer = shap.TreeExplainer(modelo)
            elif hasattr(modelo, 'feature_importances_'):  # Tree-based regressor
                explainer = shap.TreeExplainer(modelo)
            else:  # Otros modelos
                explainer = shap.KernelExplainer(modelo.predict, datos)

            shap_values = explainer.shap_values(datos)
            return {
                'values': shap_values,
                'explainer': explainer,
                'expected_value': explainer.expected_value
            }
        except Exception as e:
            self.logger.error(f"Error calculando SHAP values: {str(e)}")
            raise

    def _calcular_shap_h2o(self, modelo, datos):
        """Calcula valores SHAP para modelos H2O"""
        try:
            # Implementación específica para H2O
            contributions = modelo.predict_contributions(datos)
            return {
                'values': contributions.as_data_frame(),
                'expected_value': modelo.model_performance().mean_residual_deviance()
            }
        except Exception as e:
            self.logger.error(f"Error calculando SHAP H2O: {str(e)}")
            raise

    def calcular_shap_local(self, modelo, datos, indices):
        """Calcula valores SHAP para casos específicos"""
        try:
            shap_data = self.calcular_shap_values(modelo, datos)
            local_explanations = []
            
            for idx in indices:
                if isinstance(shap_data['values'], list):
                    # Para clasificación multiclase
                    values = [sv[idx] for sv in shap_data['values']]
                else:
                    values = shap_data['values'][idx]
                    
                local_explanations.append({
                    'caso': idx,
                    'shap_values': values,
                    'base_value': shap_data['expected_value']
                })
            
            return local_explanations
        except Exception as e:
            self.logger.error(f"Error en SHAP local: {str(e)}")
            raise

    def calcular_pdp(self, modelo, datos, objetivo, num_features=5):
        """Calcula gráficos de dependencia parcial"""
        try:
            from sklearn.inspection import partial_dependence
            
            feature_importance = self.calcular_importancia_variables(modelo, datos)
            top_features = feature_importance[:num_features]
            
            pdp_results = {}
            for feature in top_features:
                pdp = partial_dependence(
                    modelo, datos, [feature],
                    kind='average'
                )
                pdp_results[feature] = {
                    'values': pdp[1][0].tolist(),
                    'feature_values': pdp[0][0].tolist()
                }
            
            return pdp_results
        except Exception as e:
            self.logger.error(f"Error calculando PDP: {str(e)}")
            raise

    def generar_lime_explicaciones(self, modelo, datos, indices, num_features=5):
        """Genera explicaciones LIME para casos individuales"""
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=datos.values,
                feature_names=datos.columns,
                mode='classification' if hasattr(modelo, 'predict_proba') else 'regression'
            )
            
            explicaciones = []
            for idx in indices:
                exp = explainer.explain_instance(
                    datos.iloc[idx].values, 
                    modelo.predict if not hasattr(modelo, 'predict_proba') else modelo.predict_proba,
                    num_features=num_features
                )
                explicaciones.append({
                    'caso': idx,
                    'explicacion': exp.as_list(),
                    'score': exp.score
                })
                
            return explicaciones
        except Exception as e:
            self.logger.error(f"Error generando explicaciones LIME: {str(e)}")
            raise

    def generar_counterfactuals(self, modelo, datos, indices, num_cf=3):
        """Genera ejemplos contrafactuales"""
        try:
            # Configurar DiCE
            dice_data = dice_ml.Data(
                dataframe=datos,
                continuous_features=datos.select_dtypes(include=[np.number]).columns.tolist(),
                outcome_name=modelo.target_name if hasattr(modelo, 'target_name') else 'prediction'
            )
            
            dice_model = dice_ml.Model(model=modelo, backend='sklearn')
            explainer = dice_ml.Dice(dice_data, dice_model)
            
            counterfactuals = []
            for idx in indices:
                cf = explainer.generate_counterfactuals(
                    datos.iloc[idx:idx+1],
                    total_CFs=num_cf,
                    desired_class='opposite'
                )
                counterfactuals.append({
                    'caso': idx,
                    'counterfactuals': cf.cf_examples_list
                })
                
            return counterfactuals
        except Exception as e:
            self.logger.error(f"Error generando counterfactuals: {str(e)}")
            raise

    def calcular_importancia_variables(self, modelo, datos):
        """Calcula importancia de variables"""
        try:
            if hasattr(modelo, 'feature_importances_'):
                importancia = pd.DataFrame({
                    'feature': datos.columns,
                    'importance': modelo.feature_importances_
                }).sort_values('importance', ascending=False)
            elif hasattr(modelo, 'coef_'):
                importancia = pd.DataFrame({
                    'feature': datos.columns,
                    'importance': np.abs(modelo.coef_)
                }).sort_values('importance', ascending=False)
            else:
                # Usar SHAP como alternativa
                shap_values = self.calcular_shap_values(modelo, datos)
                importancia = pd.DataFrame({
                    'feature': datos.columns,
                    'importance': np.abs(shap_values['values']).mean(axis=0)
                }).sort_values('importance', ascending=False)
            
            return importancia
        except Exception as e:
            self.logger.error(f"Error calculando importancia de variables: {str(e)}")
            raise 