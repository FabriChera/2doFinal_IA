import pandas as pd
import numpy as np
import plotly.express as px

# Cargar el dataset
file_path = 'A_A1_rellenado.csv'
data = pd.read_csv(file_path, parse_dates = True, index_col='datetime')

# Calcular el Z-Score
data['z_score'] = (data['consumption'] - data['consumption'].mean()) / data['consumption'].std()

# Determinar los outliers usando un umbral de 3 desviaciones estándar
threshold = 3
data['outlier'] = data['z_score'].apply(lambda x: 1 if np.abs(x) > threshold else 0)

# Los puntos con 'outlier' == 1 son outliers
outliers = data[data['outlier'] == 1]

# Guardar el dataset modificado
output_file_path = 'A_A1_with_outliers_zscore.csv'
data.to_csv(output_file_path)

# Mostrar los primeros 10 registros para verificar
print(data.head(10))

# Mostrar los outliers detectados
print(outliers)

# Graficar el dataset y los outliers detectados
fig = px.scatter(data, x=data.index, y='consumption', color='outlier',
                 title='Consumo por Hora con Outliers Detectados',
                 labels={'consumption': 'Consumo', 'index': 'Fecha y Hora'},
                 color_discrete_map={0: 'blue', 1: 'red'})  # 0: normal, 1: outlier

fig.update_traces(marker=dict(size=5))  # Ajustar el tamaño de los puntos
fig.show()