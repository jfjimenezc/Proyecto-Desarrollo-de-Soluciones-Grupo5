# Conteo y Clasificación de Fauna en Imágenes Aéreas con Deep Learning:

Este repositorio contiene la implementación, ajustes de fine-tuning y experimentaciones con la arquitectura HerdNet, otientadas al conteo y clasificación automática de 6 especies de animales a partir de imágenes aéreas.

## Objetivo del Proyecto:

Desarrollar un modelo de Deep Learning capaz de detectar y contar animales en imágenes aéreas de manera precisa, especialmente en escenarios de oclusión, distribución densa y clases desbalanceadas.

## Estructura del repositorio:
|--notebooks/
||--experimento_0_fine_tuning_exp_2
||--experimento_0_y_fine_tuning_exp_3
||--experimento_0_y_Fine_Tuning_4
|--models/
||--herdnet_model_exp_1.pth
||--herdnet_model_exp_2_OFFICIAL.pth
||--herdnet_model_exp_3_OFFICIAL.pth
||--herdnet_model_exp_4_OFFICIAL.pth

## Data
Se utiliza el dataset oficial del artículo https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0 
- **Clases**: búfalo, elefante, kob, topi, jabalí y antílope acuático
- **División**: entrenamiento (157 imágenes), validación (111 imágenes), prueba (121 imágenes).

## Modelos:
- **Experimento 1:** 
- **Experimento 2:** Ajuste fino superficial del modelo HerdNet, en donde se congelaron todas las capas del modelo para evitar que el entrenamiento efecte los pesos de la parte más profunda de la red y solo se permitieron actualizaciones en las últimas capas del modelo, específicamente en las últimas capas convolucionales del backbone (level5) y en la capa encargada de la clasificación final fc. Adicionalmente, se hizo una reducción de la Tasa de Aprendizaje a 1e-5, más bajo que el modelo anterior con el fin de evitar sobreajuste en las capas superiores, y se redujo la penalización weight decay de 5e-4 para mejorar la estabilidad del entrenamiento. En cuanto al optimizador, se creó un optimizador Adam que afectaría únicamente las capas descongeladas, asegurando que el modelo no pierda el conocimiento previo y se mantuvo el mismo número de épocas. 
- **Experimento 3:** Fine-Tuning enfocado en la adaptación final, en donde se congelaron todas las capas del modelo para preservar los pesos previamente entrenados y se descongelaron las capas superficiales, correspondientes a los niveles level 4 y 5, y la capa de clasificación fc con el fin de especializar las salidas del modelo para el nuevo dominio. Con el objetivo de evitar el sobreajuste y permitir una adaptación controlada, se configuró una tasa de aprendizaje reducida (1e-5) y un parámetro de regularización weight decay de 5e-4. Además, se empleó una estrategia de aprendizaje diferencial con tasas distintas para cada bloque descongelado: 5e-6 para level4, 1e-5 para level5 y la capa fc. Esta segmentación permitió una refinación progresiva y controlada de los pesos. Se utilizó el optimizador AdamW y se implementó un agendador de tipo ReduceLROnPlateau, que ajusta dinámicamente la tasa de aprendizaje si la métrica de validación F1 Score deja de mejorar. El modelo fue entrenado durante 10 épocas, evaluando el desempeño en cada iteración y seleccionando como checkpoint el modelo con mejor F1 Score sobre el set de validación.  
- **Experimento 4:** Entrenamiento enfocado en capas profundas: especialización controlada del modelo HerdNet, en donde se tuvo como objetivo evaluar el desempeño del modelo HerdNet bajo condiciones controladas, replicando parcialmente el entorno propuesto en el artículo de referencia. Para ello, se aplicó un fine-tunning dirigido en las capas profundas del modelo, específicamente en level4, level5, y fc, mientras que el resto de los parámetros se mantuvieron congelados para preservar el conocimiento previamente adquirido. Con el objetivo de optimizar la especialización de las capas descongeladas, se asignaron tasas de aprendizaje diferenciadas por grupo de capas, combinadas con el optimizador AdamW y un scheduler ReduceLROnPlateay, que ajustaba dinámicamente el aprendizaje en función del rendimiento del F1 Score en validación. Las tasas utilizadas fueron: 5e-6 para el level4, 1e-5 para el level5, 5e-5 para la capa final fc y 2e-4 como weight decay global. El entrenamiento se llevó a cabo durante 10 épocas, con validación al finalizar cada una y selección automática del mejor modelo con base en el F1 Score más alto obtenido, la cual, demostró una mejora significativa en la capacidad de generalización del modelo frente a otras versiones con fine-tuning, mostrando una mayor precisión en la detección de objetos en el conjunto de prueba.

## Hallazgos relevantes:
- El fine-tuning parcial, manteniendo capas profundas congeladas, permite mejorar la especialización sin degradar el conocimiento general.
- El uso de tasas de aprendizaje diferenciadas por capa potencia la estabilidad del entrenamiento.
- Las clases con menor representación muestran menor desempeño, sugiriendo la necesidad de técnicas de balanceo u obtener más muestras de las mismas.



#########ANTERIOR VERSIÓN:#########################

Propuesta de proyecto en visión artificial


Título	Desarrollo de un modelo de deep learning para el conteo y detección de animales en manadas densas a partir de imágenes aéreas
Organización/Grupo de investigación	Proyecto Guacamaya , CINFONIA
Experto	Isai Daniel Chacon Silva. Magister en Ingeniería Biomédica, Universidad de los Andes. Investigador, Laboratorio de Microsoft AI for Good. 

1. Descripción.
Este proyecto se enmarca en el área de biodiversidad y conservación del medio ambiente, específicamente en la gestión de conflictos entre la vida silvestre y el ganado en áreas protegidas de África subsahariana. El aumento acelerado de la población humana en esta región ha generado una expansión paralela en la cantidad de ganado, lo cual trae como consecuencia conflictos por el uso de recursos como pastizales y fuentes de agua. Estos conflictos impactan negativamente tanto en la conservación de la biodiversidad como en la sostenibilidad de las comunidades rurales.
La conservación de la biodiversidad en África, además de preservar ecosistemas únicos, es clave para mitigar el cambio climático y promover la sostenibilidad económica de las comunidades locales. En este contexto, la capacidad de monitorear con precisión la densidad del ganado y la fauna silvestre es fundamental para alcanzar un equilibrio entre los objetivos de conservación y los intereses de las comunidades locales.
El conteo manual de ganado y fauna silvestre en imágenes aéreas representa un reto significativo. Métodos tradicionales, como la observación desde aeronaves o el conteo visual de fotografías, son propensos a errores, especialmente en situaciones de manadas densas, y consumen una gran cantidad de tiempo y recursos. Las arquitecturas tradicionales de redes neuronales convolucionales (CNN), si bien han mostrado potencial, presentan limitaciones para contar animales en manadas densas debido a factores como la oclusión mutua, los fondos complejos, las variaciones de escala y la distribución no uniforme de los individuos. Esto genera imprecisiones en la detección y el conteo. La falta de herramientas automáticas y confiables impide el monitoreo frecuente y eficiente, lo que limita la capacidad de tomar decisiones informadas sobre la gestión de recursos.
El impacto del proyecto será multidimensional:
•	Ambiental: Mejora de la conservación de ecosistemas al gestionar de manera sostenible los conflictos entre ganado y fauna silvestre.
•	Social: Apoyo a las comunidades rurales mediante un mejor manejo de sus recursos, minimizando los conflictos.
•	Tecnológico: Avance en el desarrollo de redes neuronales adaptadas al conteo denso en imágenes aéreas, con aplicaciones que se pueden generalizar más allá del contexto inicial.
El objetivo principal es diseñar un modelo basado en aprendizaje profundo que permita detectar y contar de manera precisa y eficiente animales en imágenes aéreas de manadas densas, considerando los desafíos inherentes al problema (oclusiones, fondos complejos, variaciones de escala). Esto sentará las bases para un monitoreo más eficiente y una gestión sostenible de la biodiversidad y los recursos naturales.
2. Conjunto de datos (dataset).
Los datos disponibles para este proyecto son públicos y accesibles a través del enlace: https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0. Esta base de datos está diseñada específicamente para tareas de conteo de animales en manadas densas utilizando imágenes aéreas y ya se encuentra organizada en tres conjuntos: entrenamiento (train), validación (val) y prueba (test). A continuación, se muestran algunas imágenes de este dataset:


Los datos están bajo la licencia CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International). Esto implica las siguientes condiciones:
•	Atribución. Se debe dar el crédito adecuado al autor original, incluir un enlace a la licencia y señalar si se realizaron modificaciones.
•	No Comercial. Los datos no pueden ser utilizados con fines comerciales.
•	Compartir Igual. Si los estudiantes generan nuevos trabajos derivados, estos deben ser distribuidos bajo la misma licencia.
•	No contiene información sensible o confidencial. Los datos son completamente públicos y adecuados para el uso académico y de investigación.
3. Validación de resultados.
Se evaluará la capacidad del modelo para detectar y contar con precisión el número total de animales en cada imagen, independientemente de la especie, así como la capacidad del modelo para clasificar correctamente a los animales detectados entre las especies disponibles: búfalo, elefante, kob, topi, facocero (warthog) y waterbuck. Para este propósito, se utilizarán métricas clave que permitan medir su efectividad tanto en el conteo de animales como en la clasificación de especies. Los criterios específicos incluyen Precisión, Recall, F1-Score, MAE, RMSE, AC (Average Confusion). Además, se utilizarán los resultados de HerdNet  como punto de comparación para evaluar el desempeño del modelo desarrollado por el grupo. HerdNet, una implementación previa de referencia, obtuvo los siguientes resultados:
F1-Score (%)	MAE	RMSE	AC (Confusión Promedio) (%)
83.5	1.9	3.6	7.8
5. Material de apoyo
•	From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning? https://www.sciencedirect.com/science/article/pii/S092427162300031X?via%3Dihub 
•	Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html
•	Deep Layer Aggregation: https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.html
También se pueden revisar el siguiente repositorio en Github:
https://github.com/Alexandre-Delplanque/HerdNet.git
Además, HerdNet tiene una implementación para hacer inferencia en el siguiente repositorio:
https://github.com/microsoft/CameraTraps/blob/main/demo/image_detection_demo_herdnet.ipynb
