#Valentina Dominguez
#Fabrizio Chera
import pandas as pd

# Cargar el dataset
file_path = 'output_data\A_A1.csv'  # Nombre del archivo CSV en la misma carpeta
data = pd.read_csv(file_path)

# Realizar la interpolación lineal en la columna 'consumption'
data['consumption'] = data['consumption'].interpolate(method='linear')

# Si el primer valor de la columna 'consumption' es NaN, rellenarlo con el promedio de los siguientes valores
if pd.isna(data['consumption'].iloc[0]):
    siguientes_valores = data['consumption'].dropna().iloc[:4]  # Tomar los siguientes 10 valores no nulos
    promedio = siguientes_valores.mean() if not siguientes_valores.empty else 0
    data['consumption'].iloc[0] = promedio

# Guardar el dataset modificado (opcional)
output_file_path = 'A_A1_rellenado.csv'  # Nombre del archivo de salida
data.to_csv(output_file_path, index=False)