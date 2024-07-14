import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos
data_path = 'output_data/A_A1.csv'
data = pd.read_csv(data_path)

# Convertir la columna datetime a formato datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Extraer caracteristicas temporales
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek

# Eliminar datos faltantes
data_clean = data.dropna(subset=['consumption'])

# Definir características (X) y la variable objetivo (y)
features = ['year', 'month', 'day', 'hour', 'day_of_week']
X = data_clean[features]
y = data_clean['consumption']

# conjunto de entrenamiento (2017-2019) y de prueba (2020)
X_train = X[X['year'] < 2020]
y_train = y[X['year'] < 2020]
X_test = X[X['year'] == 2020]
y_test = y[X['year'] == 2020]

# Obtener las fechas correspondientes para el conjunto de prueba
datetime_test = data_clean['datetime'][X['year'] == 2020]

# Crear y entrenar el modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calcular errores absolutos
errors = np.abs(y_test - y_pred_test)

# Evaluar el modelo en el conjunto de entrenamiento
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Evaluar el modelo en el conjunto de prueba
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("MSE (entrenamiento):", mse_train)
print("MAE (entrenamiento):", mae_train)
print("R^2 (entrenamiento):", r2_train)
print("\nMSE (prueba):", mse_test)
print("MAE (prueba):", mae_test)
print("R^2 (prueba):", r2_test)

# Crear histograma de errores absolutos
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=20, edgecolor='black', color='purple')
plt.xlabel('Error absoluto')
plt.ylabel('Frecuencia')
plt.title('Histograma de Errores Absolutos de Predicción')
plt.grid(True)
plt.show()

# Visualizar los resultados con colores personalizados para todos los datos de prueba
plt.figure(figsize=(14, 7))
plt.plot(datetime_test, y_test.values, color='purple', label='Valores reales')
plt.plot(datetime_test, y_pred_test, color='pink', label='Predicciones')
plt.legend()
plt.xlabel('Fecha y Hora')
plt.ylabel('Consumo (A)')
plt.title('Predicción del Consumo por Hora con Random Forest (Todos los datos de prueba)')
plt.xticks(rotation=45)
plt.show()

# Filtrar los datos para una semana específica
start_date = '2020-01-01'
end_date = '2020-01-07'
mask = (datetime_test >= start_date) & (datetime_test <= end_date)
datetime_test_week = datetime_test[mask]
y_test_week = y_test[mask]
y_pred_test_week = y_pred_test[mask]

# Visualizar los resultados con colores personalizados para la semana específica
plt.figure(figsize=(14, 7))
plt.plot(datetime_test_week, y_test_week.values, color='purple', label='Valores reales')
plt.plot(datetime_test_week, y_pred_test_week, color='pink', label='Predicciones')
plt.legend()
plt.xlabel('Fecha y Hora')
plt.ylabel('Consumo (A)')
plt.title('Predicción del Consumo por Hora con Random Forest para la primera semana de enero 2020')
plt.xticks(rotation=45)
plt.show()