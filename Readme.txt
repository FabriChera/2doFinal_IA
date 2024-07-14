El repositorio cuenta con con tres datasets 'A_A1', 'electricity-consumption-processed', 'meteorological-processed'
Respectivamente corresponde a 1) El consumo de corriente de la linea 1 de la subsestacion A 2) El consumo de
corriente de cada linea de cada subestacion y por ultimo los datos meteorologicos de la region.

La carpeta 'output_data' es lo que devuelve el script '2doparcial_ExportacionDeDatos' que separa cada linea de cada subestacion

La carpeta 'Preprocesamiento(erroneo)' es una parte de los intentos que tuvimos de procesar los datos del dataset sin procesar

El notebook 'PrediccionEnergiaSubV1.0' corresponde a la red LSTM teniendo como entrada 'A_A1' y 'meteorological-processed'

El script 'RegresionLineal' corresponde a las predicciones hechas con Regresion Lineal recibiendo el script como entrada 'A_A1'

El script 'RandomForest' corresponde a las predicciones hechas con Random Forest recibiendo el script como entrada 'A_A1'