import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px

# Cargar el dataset
file_path = 'A_A1_rellenado.csv'
data = pd.read_csv(file_path)

# Asegurarse de que la columna 'timestamp' sea de tipo datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Visualizar los datos de consumo
plt.figure(figsize=(10, 6))
plt.plot(data['datetime'], data['consumption'], label='Consumo')
plt.title('Consumo por Hora')
plt.xlabel('Fecha')
plt.ylabel('Consumo')
plt.legend()
plt.show()

# Prueba de Dickey-Fuller Aumentada (ADF) para verificar la estacionariedad
result = adfuller(data['consumption'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Diferenciación si no es estacionaria
data['consumption_diff'] = data['consumption'].diff().dropna()

# Graficar ACF y PACF
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
acf_values = acf(data['consumption_diff'].dropna(), nlags=40)
pacf_values = pacf(data['consumption_diff'].dropna(), nlags=40)
ax[0].stem(acf_values)
ax[0].set_title('ACF')
ax[1].stem(pacf_values)
ax[1].set_title('PACF')
plt.show()

# Ajuste del modelo SARIMA
order = (1, 1, 1)  # (p, d, q)
seasonal_order = (1, 1, 1, 24)  # (P, D, Q, S)

model = SARIMAX(data['consumption'], order=order, seasonal_order=seasonal_order)
results = model.fit()

# Calcular los residuos
data['residuals'] = results.resid

# Determinar los outliers usando un umbral de 3 desviaciones estándar
threshold = 3
data['outlier'] = data['residuals'].apply(lambda x: 1 if np.abs(x) > threshold * data['residuals'].std() else 0)

# Los puntos con 'outlier' == 1 son outliers
outliers = data[data['outlier'] == 1]

# Guardar el dataset modificado (opcional)
output_file_path = 'A_A1_with_outliers_sarima.csv'
data.to_csv(output_file_path, index=False)

# Mostrar los primeros 10 registros para verificar
print(data.head(10))

# Mostrar los outliers detectados
print(outliers)

# Graficar el dataset y los outliers detectados
fig = px.scatter(data, x=data.index, y='consumption', color='outlier',
                 title='Consumo por Hora con Outliers Detectados (SARIMA)',
                 labels={'consumption': 'Consumo', 'index': 'Timestamp'},
                 color_discrete_map={0: 'blue', 1: 'red'})  # 0: normal, 1: outlier

fig.update_traces(marker=dict(size=5))  # Ajustar el tamaño de los puntos
fig.show()