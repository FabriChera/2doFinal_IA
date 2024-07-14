import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data_path = 'output_data/A_A1.csv'
data = pd.read_csv(data_path)

# Convertir la columna datetime a formato datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Ordenar los datos por datetime
data = data.sort_values('datetime')

# Crear los lags
def create_lags(df, lag=1):
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['consumption'].shift(i)
    return df

# Elegir el cantidad de lags
num_lags = 24  # Lags de 24 horas

# Crear características de lag
data = create_lags(data, lag=num_lags)

# Eliminar filas con valores faltantes
data = data.dropna()

# Extraer caracteristicas temporales adicionales
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek

# Definir características (X) y la variable objetivo (y)
features = [f'lag_{i}' for i in range(1, num_lags + 1)] + ['year', 'month', 'day', 'hour', 'day_of_week']
X = data[features]
y = data['consumption']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train = X[data['datetime'].dt.year < 2020]
y_train = y[data['datetime'].dt.year < 2020]
X_test = X[data['datetime'].dt.year == 2020]
y_test = y[data['datetime'].dt.year == 2020]

# Crear y entrenar el modelo de regresion lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred_train = model.predict(X_train)

# Calcular errores de entrenamiento
train_errors = np.abs(y_train - y_pred_train)

# Evaluar el modelo en el conjunto de entrenamiento
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print("MSE (entrenamiento):", mse_train)
print("MAE (entrenamiento):", mae_train)
print("R^2 (entrenamiento):", r2_train)

# Graficar los resultados en el conjunto de entrenamiento
plt.figure(figsize=(14, 7))
plt.plot(data[data['datetime'].dt.year < 2020]['datetime'], y_train.values, color='purple', label='Valores reales')
plt.plot(data[data['datetime'].dt.year < 2020]['datetime'], y_pred_train, color='pink', label='Predicciones')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Consumo (A)')
plt.title('Prediccion del Consumo por Hora (Entrenamiento)')
plt.show()

# Crear histograma de errores absolutos de entrenamiento
plt.figure(figsize=(10, 6))
plt.hist(train_errors, bins=20, edgecolor='black', color='purple', label='Errores Entrenamiento')
plt.xlabel('Error absoluto')
plt.ylabel('Frecuencia')
plt.title('Histograma de Errores Absolutos de Predicción (Entrenamiento)')
plt.legend()
plt.grid(True)
plt.show()

# Hacer predicciones en el conjunto de prueba
y_pred_test = model.predict(X_test)

# Calcular errores de prueba
test_errors = np.abs(y_test - y_pred_test)

# Evaluar el modelo en el conjunto de prueba
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("MSE (Prueba):", mse_test)
print("MAE (Prueba):", mae_test)
print("R^2 (Prueba):", r2_test)

# Visualizar los resultados en el conjunto de prueba
plt.figure(figsize=(14, 7))
plt.plot(data[data['datetime'].dt.year == 2020]['datetime'], y_test.values, color='purple', label='Valores reales (prueba)')
plt.plot(data[data['datetime'].dt.year == 2020]['datetime'], y_pred_test, color='pink', label='Predicciones (prueba)')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Consumo (A)')
plt.title('Prediccion del Consumo por Hora (Prueba)')
plt.show()

# Crear histograma de errores absolutos de prueba
plt.figure(figsize=(10, 6))
plt.hist(test_errors, bins=20, edgecolor='black', color='purple', label='Errores Prueba')
plt.xlabel('Error absoluto')
plt.ylabel('Frecuencia')
plt.title('Histograma de Errores Absolutos de Prediccion (Prueba)')
plt.legend()
plt.grid(True)
plt.show()

# Filtrar los datos para una semana especfica en el conjunto de prueba
start_date = '2020-01-01'
end_date = '2020-01-07'
mask = (data['datetime'] >= start_date) & (data['datetime'] <= end_date)
data_week = data[mask]

X_test_week = data_week[features]
y_test_week = data_week['consumption']
y_pred_test_week = model.predict(X_test_week)

# Visualizar los resultados para la semana especifica
plt.figure(figsize=(14, 7))
plt.plot(data_week['datetime'], y_test_week.values, color='purple', label='Valores reales')
plt.plot(data_week['datetime'], y_pred_test_week, color='pink', label='Predicciones')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Consumo (A)')
plt.title('Prediccion del Consumo por Hora (Primera semana de enero 2020)')
plt.xticks(rotation=45)
plt.show()