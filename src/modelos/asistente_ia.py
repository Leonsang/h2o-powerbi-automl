import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class AsistenteIA:
    """
    Asistente IA para generar explicaciones en lenguaje natural de los resultados del modelo.
    """
    
    def __init__(self):
        self.prompts = {
            'analisis_dataset': """
            Basado en el análisis del dataset, puedo decir que:
            - El conjunto de datos tiene {n_registros} registros y {n_features} características
            - {valores_nulos_desc}
            - Es un problema de {tipo_problema}
            - {balance_desc}
            
            Recomendaciones:
            {recomendaciones}
            """,
            
            'metricas_modelo': """
            El modelo muestra el siguiente rendimiento:
            - Precisión global: {precision}%
            - AUC: {auc} ({auc_desc})
            - Tasa de error: {error_rate}%
            
            Esto significa que {interpretacion_rendimiento}
            """,
            
            'importancia_variables': """
            Las variables más importantes para predecir {objetivo} son:
            {variables_importantes}
            
            Esto sugiere que {interpretacion_importancia}
            """,
            
            'prediccion_individual': """
            Para este caso específico:
            - La predicción es: {prediccion}
            - Nivel de confianza: {confianza}%
            
            Los factores más influyentes fueron:
            {factores_principales}
            
            {interpretacion_prediccion}
            """,
            
            'analisis_errores': """
            Análisis de errores del modelo:
            - Falsos positivos: {falsos_positivos}%
            - Falsos negativos: {falsos_negativos}%
            
            Patrones identificados:
            {patrones_error}
            
            Recomendaciones para mejorar:
            {recomendaciones_mejora}
            """,
            
            'analisis_sesgos': """
            Análisis de sesgos del modelo:
            {sesgos_identificados}
            
            Impacto en grupos específicos:
            {impacto_grupos}
            
            Recomendaciones para mitigar sesgos:
            {recomendaciones_sesgo}
            """,
            
            'reporte_completo': """
            REPORTE DE ANÁLISIS DEL MODELO
            Fecha: {fecha}
            
            1. RESUMEN EJECUTIVO
            {resumen_ejecutivo}
            
            2. ANÁLISIS DEL DATASET
            {analisis_dataset}
            
            3. RENDIMIENTO DEL MODELO
            {metricas_modelo}
            
            4. INTERPRETABILIDAD
            {interpretabilidad}
            
            5. ANÁLISIS DE ERRORES Y SESGOS
            {analisis_errores_sesgos}
            
            6. RECOMENDACIONES
            {recomendaciones}
            
            7. SIGUIENTES PASOS
            {siguientes_pasos}
            """
        }

    def generar_reporte_completo(self, resultados: Dict) -> str:
        """Genera un reporte completo del modelo"""
        try:
            # Obtener todos los análisis
            analisis_dataset = self.explicar_analisis_dataset(resultados['analisis_dataset'])
            metricas = self.explicar_metricas(resultados['metricas'])
            importancia = self.explicar_importancia_variables(resultados['interpretacion'], 
                                                           resultados['objetivo'])
            errores = self.analizar_errores(resultados['metricas'])
            sesgos = self.analizar_sesgos(resultados)
            
            # Generar resumen ejecutivo
            resumen = self._generar_resumen_ejecutivo(resultados)
            
            # Generar recomendaciones
            recomendaciones = self._generar_recomendaciones(resultados)
            
            # Generar siguientes pasos
            siguientes_pasos = self._recomendar_siguientes_pasos(resultados)
            
            return self.prompts['reporte_completo'].format(
                fecha=datetime.now().strftime("%Y-%m-%d %H:%M"),
                resumen_ejecutivo=resumen,
                analisis_dataset=analisis_dataset,
                metricas_modelo=metricas,
                interpretabilidad=importancia,
                analisis_errores_sesgos=f"{errores}\n\n{sesgos}",
                recomendaciones=recomendaciones,
                siguientes_pasos=siguientes_pasos
            )
        except Exception as e:
            logger.error(f"Error generando reporte completo: {str(e)}")
            return "No se pudo generar el reporte completo"

    def analizar_errores(self, metricas: Dict) -> str:
        """Analiza los errores del modelo en detalle"""
        try:
            if 'clasificacion' in metricas:
                conf_matrix = metricas['clasificacion']['confusion_matrix']
                
                # Calcular tasas de error
                fp = float(conf_matrix['Error'][0].split()[0]) * 100
                fn = float(conf_matrix['Error'][1].split()[0]) * 100
                
                # Identificar patrones en errores
                patrones = self._identificar_patrones_error(conf_matrix)
                
                # Generar recomendaciones
                recomendaciones = self._generar_recomendaciones_error(fp, fn, patrones)
                
                return self.prompts['analisis_errores'].format(
                    falsos_positivos=f"{fp:.1f}",
                    falsos_negativos=f"{fn:.1f}",
                    patrones_error=patrones,
                    recomendaciones_mejora=recomendaciones
                )
            else:
                return "Análisis de errores no disponible para este tipo de modelo"
        except Exception as e:
            logger.error(f"Error en análisis de errores: {str(e)}")
            return "No se pudo realizar el análisis de errores"

    def analizar_sesgos(self, resultados: Dict) -> str:
        """Analiza posibles sesgos en el modelo"""
        try:
            # Identificar sesgos en variables sensibles
            sesgos = self._identificar_sesgos(resultados)
            
            # Analizar impacto en grupos específicos
            impacto = self._analizar_impacto_grupos(resultados)
            
            # Generar recomendaciones
            recomendaciones = self._generar_recomendaciones_sesgo(sesgos, impacto)
            
            return self.prompts['analisis_sesgos'].format(
                sesgos_identificados=sesgos,
                impacto_grupos=impacto,
                recomendaciones_sesgo=recomendaciones
            )
        except Exception as e:
            logger.error(f"Error en análisis de sesgos: {str(e)}")
            return "No se pudo realizar el análisis de sesgos"

    def _generar_resumen_ejecutivo(self, resultados: Dict) -> str:
        """Genera un resumen ejecutivo del análisis"""
        try:
            metricas = resultados['metricas']
            if 'clasificacion' in metricas:
                precision = (1 - float(metricas['clasificacion']['confusion_matrix']['Error'][2].split()[0])) * 100
                rendimiento = f"precisión del {precision:.1f}%"
            else:
                r2 = metricas['basic']['r2'] * 100
                rendimiento = f"R² del {r2:.1f}%"
            
            # Obtener top 3 variables importantes
            vars_imp = sorted(
                resultados['interpretacion']['importancia_variables'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_vars = [var for var, _ in vars_imp[:3]]
            
            return f"""
            El modelo entrenado para {resultados['objetivo']} muestra un {rendimiento}.
            Las principales variables que influyen en las predicciones son {', '.join(top_vars)}.
            El modelo es adecuado para su uso en {self._identificar_casos_uso(resultados)}.
            """
        except Exception as e:
            logger.error(f"Error generando resumen ejecutivo: {str(e)}")
            return "No se pudo generar el resumen ejecutivo"

    def _generar_recomendaciones_error(self, fp: float, fn: float, patrones: str) -> str:
        """Genera recomendaciones basadas en los errores del modelo"""
        try:
            recomendaciones = []
            
            # Analizar falsos positivos y negativos
            if fp > fn:
                recomendaciones.append(
                    "Considerar ajustar el umbral de clasificación para reducir falsos positivos"
                )
            else:
                recomendaciones.append(
                    "Considerar ajustar el umbral de clasificación para reducir falsos negativos"
                )
            
            # Agregar recomendaciones basadas en patrones
            if "falsos positivos" in patrones.lower():
                recomendaciones.append(
                    "Revisar casos de falsos positivos para identificar patrones comunes"
                )
            if "falsos negativos" in patrones.lower():
                recomendaciones.append(
                    "Revisar casos de falsos negativos para identificar patrones comunes"
                )
            
            recomendaciones.extend([
                "Evaluar la calidad de los datos en los casos mal clasificados",
                "Considerar agregar más features relevantes",
                "Explorar técnicas de balanceo de clases si es necesario"
            ])
            
            return "\n".join(f"- {r}" for r in recomendaciones)
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones de error: {str(e)}")
            return "No se pudieron generar recomendaciones"

    def _generar_recomendaciones_sesgo(self, sesgos: List[str], impacto: List[str]) -> str:
        """Genera recomendaciones para mitigar sesgos"""
        try:
            recomendaciones = []
            
            # Recomendaciones basadas en sesgos detectados
            if sesgos:
                recomendaciones.append(
                    "Evaluar y ajustar el proceso de recolección de datos para reducir sesgos"
                )
                recomendaciones.append(
                    "Considerar técnicas de muestreo estratificado"
                )
            
            # Recomendaciones basadas en impacto
            if impacto:
                recomendaciones.append(
                    "Monitorear el rendimiento del modelo en diferentes grupos demográficos"
                )
                recomendaciones.append(
                    "Implementar métricas de equidad en la evaluación del modelo"
                )
            
            # Recomendaciones generales
            recomendaciones.extend([
                "Documentar y comunicar los sesgos identificados",
                "Establecer un proceso de revisión periódica de sesgos",
                "Considerar el uso de técnicas de debiasing si es necesario"
            ])
            
            return "\n".join(f"- {r}" for r in recomendaciones)
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones de sesgo: {str(e)}")
            return "No se pudieron generar recomendaciones"

    def _generar_recomendaciones(self, resultados: Dict) -> str:
        """Genera recomendaciones basadas en todos los análisis"""
        try:
            recomendaciones = []
            
            # Recomendaciones de datos
            if 'valores_nulos' in resultados['analisis_dataset']['estadisticas_basicas']:
                recomendaciones.append("Mejorar la calidad de los datos reduciendo valores faltantes")
            
            # Recomendaciones de rendimiento
            if 'clasificacion' in resultados['metricas']:
                precision = (1 - float(resultados['metricas']['clasificacion']['confusion_matrix']['Error'][2].split()[0])) * 100
                if precision < 80:
                    recomendaciones.append("Explorar técnicas de optimización del modelo")
            
            # Recomendaciones de interpretabilidad
            if len(resultados['interpretacion']['importancia_variables']) > 10:
                recomendaciones.append("Considerar reducción de dimensionalidad")
            
            return "\n".join(f"- {r}" for r in recomendaciones)
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {str(e)}")
            return "No se pudieron generar recomendaciones"

    def _recomendar_siguientes_pasos(self, resultados: Dict) -> str:
        """Recomienda los siguientes pasos a seguir"""
        try:
            pasos = []
            
            # Pasos de mejora de datos
            if resultados['analisis_dataset']['recomendaciones']:
                pasos.append("1. Implementar mejoras en la calidad de los datos")
            
            # Pasos de optimización
            if 'clasificacion' in resultados['metricas']:
                precision = (1 - float(resultados['metricas']['clasificacion']['confusion_matrix']['Error'][2].split()[0])) * 100
                if precision < 90:
                    pasos.append("2. Realizar optimización de hiperparámetros")
            
            # Pasos de monitoreo
            pasos.append("3. Implementar sistema de monitoreo del modelo")
            
            return "\n".join(pasos)
        except Exception as e:
            logger.error(f"Error recomendando siguientes pasos: {str(e)}")
            return "No se pudieron recomendar siguientes pasos"

    def _identificar_patrones_error(self, conf_matrix: Dict) -> str:
        """Identifica patrones en los errores del modelo"""
        try:
            patrones = []
            
            # Analizar distribución de errores
            fp = float(conf_matrix['Error'][0].split()[0])
            fn = float(conf_matrix['Error'][1].split()[0])
            
            if fp > fn:
                patrones.append("El modelo tiende a generar más falsos positivos")
            elif fn > fp:
                patrones.append("El modelo tiende a generar más falsos negativos")
            
            return "\n".join(f"- {p}" for p in patrones)
        except Exception as e:
            logger.error(f"Error identificando patrones: {str(e)}")
            return "No se pudieron identificar patrones"

    def _identificar_sesgos(self, resultados: Dict) -> str:
        """Identifica posibles sesgos en el modelo"""
        try:
            sesgos = []
            
            # Analizar variables sensibles
            variables_sensibles = ['Sex', 'Age', 'Pclass']  # Ejemplo para Titanic
            for var in variables_sensibles:
                if var in resultados['interpretacion']['importancia_variables']:
                    sesgos.append(f"Posible sesgo en {var}")
            
            return "\n".join(f"- {s}" for s in sesgos)
        except Exception as e:
            logger.error(f"Error identificando sesgos: {str(e)}")
            return "No se pudieron identificar sesgos"

    def _analizar_impacto_grupos(self, resultados: Dict) -> str:
        """Analiza el impacto del modelo en diferentes grupos"""
        try:
            impactos = []
            
            # Analizar rendimiento por grupo
            if 'grupos_analisis' in resultados:
                for grupo, metricas in resultados['grupos_analisis'].items():
                    impactos.append(f"Grupo {grupo}: {metricas['rendimiento']}")
            
            return "\n".join(f"- {i}" for i in impactos)
        except Exception as e:
            logger.error(f"Error analizando impacto: {str(e)}")
            return "No se pudo analizar el impacto"

    def _identificar_casos_uso(self, resultados: Dict) -> str:
        """Identifica casos de uso apropiados para el modelo"""
        try:
            casos = []
            
            # Basado en rendimiento
            if 'clasificacion' in resultados['metricas']:
                precision = (1 - float(resultados['metricas']['clasificacion']['confusion_matrix']['Error'][2].split()[0])) * 100
                if precision > 90:
                    casos.append("decisiones críticas")
                elif precision > 80:
                    casos.append("decisiones de negocio generales")
                else:
                    casos.append("apoyo a decisiones con supervisión humana")
            
            return ", ".join(casos)
        except Exception as e:
            logger.error(f"Error identificando casos de uso: {str(e)}")
            return "No se pudieron identificar casos de uso"

    def explicar_analisis_dataset(self, analisis: dict) -> str:
        """Genera explicación en lenguaje natural del análisis del dataset"""
        try:
            stats = analisis['estadisticas_basicas']
            
            # Descripción de valores nulos
            nulos = stats['valores_nulos']
            if any(nulos.values()):
                cols_nulos = [k for k, v in nulos.items() if v > 0]
                valores_nulos_desc = f"Se encontraron valores nulos en las columnas: {', '.join(cols_nulos)}"
            else:
                valores_nulos_desc = "No se encontraron valores nulos"
            
            # Descripción del tipo de problema
            tipo_problema = analisis['analisis_objetivo']['tipo']
            
            # Descripción del balance
            if analisis['analisis_desbalanceo']:
                ratio = analisis['analisis_desbalanceo']['ratio_desbalanceo']
                if ratio < 0.2:
                    balance_desc = f"Dataset desbalanceado (ratio: {ratio:.2f})"
                else:
                    balance_desc = "Dataset balanceado"
            else:
                balance_desc = "No aplica análisis de balance"
            
            # Recomendaciones
            recomendaciones = "\n".join([f"- {r['descripcion']}" for r in analisis['recomendaciones']])
            
            return self.prompts['analisis_dataset'].format(
                n_registros=stats['n_registros'],
                n_features=stats['n_features'],
                valores_nulos_desc=valores_nulos_desc,
                tipo_problema=tipo_problema,
                balance_desc=balance_desc,
                recomendaciones=recomendaciones
            )
        except Exception as e:
            logger.error(f"Error explicando análisis de dataset: {str(e)}")
            return "No se pudo generar la explicación del análisis del dataset"

    def explicar_metricas(self, metricas: dict) -> str:
        """Genera explicación en lenguaje natural de las métricas del modelo"""
        try:
            if 'clasificacion' in metricas:
                precision = (1 - float(metricas['clasificacion']['confusion_matrix']['Error'][2].split()[0])) * 100
                auc = metricas['clasificacion']['auc']
                error_rate = float(metricas['clasificacion']['confusion_matrix']['Error'][2].split()[0]) * 100
                
                # Descripción del AUC
                if auc > 0.9:
                    auc_desc = "excelente"
                elif auc > 0.8:
                    auc_desc = "bueno"
                else:
                    auc_desc = "regular"
                
                # Interpretación del rendimiento
                if precision > 90:
                    interpretacion = "el modelo tiene un rendimiento excelente"
                elif precision > 80:
                    interpretacion = "el modelo tiene un buen rendimiento"
                else:
                    interpretacion = "el modelo tiene un rendimiento que podría mejorarse"
                
            else:
                precision = metricas['basic']['r2'] * 100
                auc = "N/A"
                auc_desc = "no aplica"
                error_rate = (1 - metricas['basic']['r2']) * 100
                interpretacion = f"el modelo explica el {precision:.1f}% de la varianza"
            
            return self.prompts['metricas_modelo'].format(
                precision=precision,
                auc=auc,
                auc_desc=auc_desc,
                error_rate=error_rate,
                interpretacion_rendimiento=interpretacion
            )
        except Exception as e:
            logger.error(f"Error explicando métricas: {str(e)}")
            return "No se pudo generar la explicación de las métricas"

    def explicar_importancia_variables(self, interpretacion: dict, objetivo: str) -> str:
        """Genera explicación en lenguaje natural de la importancia de variables"""
        try:
            # Obtener top 5 variables más importantes
            vars_imp = sorted(
                interpretacion['importancia_variables'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            variables_importantes = "\n".join([
                f"- {var}: {imp:.3f}" for var, imp in vars_imp
            ])
            
            # Generar interpretación
            interpretacion_importancia = "las variables seleccionadas son las que más influyen en la predicción"
            if len(vars_imp) > 0:
                var_principal = vars_imp[0][0]
                interpretacion_importancia = f"la variable {var_principal} es la más influyente en la predicción"
            
            return self.prompts['importancia_variables'].format(
                objetivo=objetivo,
                variables_importantes=variables_importantes,
                interpretacion_importancia=interpretacion_importancia
            )
        except Exception as e:
            logger.error(f"Error explicando importancia de variables: {str(e)}")
            return "No se pudo generar la explicación de la importancia de variables" 