from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd
import json
from src.logger import Logger
from src.modelo_manager_ia import ModeloManagerIA

logger = Logger('asistente_ia')

class AsistenteDataScience:
    def __init__(self, modelo_base='orca-mini'):
        """Inicializa el asistente de IA"""
        try:
            # Gestionar modelo
            self.modelo_manager = ModeloManagerIA(modelo_base)
            model_path = self.modelo_manager.get_model_path()
            
            # Configurar modelo
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = GPT4All(
                model=model_path,
                callback_manager=callback_manager,
                verbose=True
            )
            
            # Template para análisis
            self.template_analisis = """
            Actúa como un Data Scientist experto analizando los siguientes resultados de un modelo:

            MÉTRICAS DEL MODELO:
            {metricas}

            IMPORTANCIA DE VARIABLES:
            {importancia_variables}

            INSIGHTS DETECTADOS:
            {insights}

            Por favor, proporciona:
            1. Un análisis profesional del rendimiento del modelo
            2. Interpretación de las variables más importantes
            3. Recomendaciones accionables basadas en los insights
            4. Posibles áreas de mejora
            5. Conclusiones generales

            Usa un lenguaje técnico pero comprensible y estructura tu respuesta.
            """
            
            self.prompt = PromptTemplate(
                input_variables=["metricas", "importancia_variables", "insights"],
                template=self.template_analisis
            )
            
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            logger.info("Asistente IA inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando asistente IA: {str(e)}")
            raise

    def interpretar_resultados(self, resultados):
        """Genera una interpretación profesional de los resultados"""
        try:
            # Formatear métricas para el prompt
            metricas_str = json.dumps(resultados['metricas'], indent=2)
            importancia_str = pd.DataFrame(resultados['importancia_variables']).to_string()
            insights_str = "\n".join([i['mensaje'] for i in resultados['insights']])
            
            # Generar análisis
            analisis = self.chain.run({
                "metricas": metricas_str,
                "importancia_variables": importancia_str,
                "insights": insights_str
            })
            
            logger.info("Análisis IA generado exitosamente")
            return analisis
            
        except Exception as e:
            logger.error(f"Error en interpretación IA: {str(e)}")
            raise

    def generar_recomendaciones_tecnicas(self, resultados):
        """Genera recomendaciones técnicas específicas"""
        try:
            template_tecnico = """
            Como Data Scientist experto, analiza estos resultados técnicos:
            
            MÉTRICAS: {metricas}
            SHAP VALUES: {shap_values}
            
            Proporciona recomendaciones técnicas específicas sobre:
            1. Ajustes de hiperparámetros
            2. Feature engineering sugerido
            3. Técnicas de modelado alternativas
            4. Validación adicional recomendada
            """
            
            prompt_tecnico = PromptTemplate(
                input_variables=["metricas", "shap_values"],
                template=template_tecnico
            )
            
            chain_tecnico = LLMChain(llm=self.llm, prompt=prompt_tecnico)
            
            recomendaciones = chain_tecnico.run({
                "metricas": json.dumps(resultados['metricas'], indent=2),
                "shap_values": str(resultados['shap_values'])
            })
            
            logger.info("Recomendaciones técnicas generadas")
            return recomendaciones
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones técnicas: {str(e)}")
            raise

    def explicar_predicciones(self, caso, explicacion_lime, counterfactual):
        """Explica predicciones específicas en lenguaje natural"""
        try:
            template_prediccion = """
            Explica la siguiente predicción de manera clara y profesional:
            
            CASO ANALIZADO:
            {caso}
            
            EXPLICACIÓN LIME:
            {lime}
            
            COUNTERFACTUAL:
            {counterfactual}
            
            Proporciona:
            1. Explicación principal de la predicción
            2. Factores más influyentes
            3. Qué cambios producirían un resultado diferente
            4. Recomendaciones específicas para este caso
            """
            
            prompt_prediccion = PromptTemplate(
                input_variables=["caso", "lime", "counterfactual"],
                template=template_prediccion
            )
            
            chain_prediccion = LLMChain(llm=self.llm, prompt=prompt_prediccion)
            
            explicacion = chain_prediccion.run({
                "caso": str(caso),
                "lime": str(explicacion_lime),
                "counterfactual": str(counterfactual)
            })
            
            logger.info(f"Explicación generada para caso {caso}")
            return explicacion
            
        except Exception as e:
            logger.error(f"Error explicando predicción: {str(e)}")
            raise 