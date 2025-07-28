import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('./housing.csv') 
print(df.head())


datos = df.dropna()
print(datos['ocean_proximity'].value_counts())
dummies = pd.get_dummies(datos['ocean_proximity'], dtype=int)
datos = pd.concat([datos, dummies], axis=1)
datos.drop('ocean_proximity', axis=1, inplace=True)

datos = datos[datos['housing_median_age'] < 50]
datos = datos[datos['median_house_value'] < 500000]
datos = datos[datos['median_income'] < 15]
datos['bedrooms_per_room'] = datos['total_bedrooms'] / datos['total_rooms']
datos['households_per_population'] = datos['households'] / datos['population']
datos['rooms_per_household'] = datos['total_rooms'] / datos['households']

datos['log_median_income'] = np.log1p(datos['median_income'])
datos['log_median_house_value'] = np.log1p(datos['median_house_value'])

print("Correlaciones con median_house_value:")
print(datos.corr()['median_house_value'].sort_values(ascending=False))
datos['room_ratio'] = datos['total_rooms'] / datos['total_bedrooms']


X = datos.drop(columns='median_house_value', axis=1) 
y = datos['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Escalado de datos (necesario para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = svm.SVR(gamma=0.001, C=100)
modelo.fit(X_train_scaled, y_train)
predicciones = modelo.predict(X_test_scaled)

mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)
mae = mean_absolute_error(y_test, predicciones)

print(f'Error (MSE): {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Visualización de predicciones vs valores reales
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, predicciones, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('SVM - Predicciones vs Valores Reales')

plt.subplot(1, 2, 2)
residuos = y_test - predicciones
plt.scatter(predicciones, residuos, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('SVM - Gráfico de Residuos')

plt.tight_layout()
plt.show()