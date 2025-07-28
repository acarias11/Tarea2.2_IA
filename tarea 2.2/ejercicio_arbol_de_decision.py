import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

modelo = DecisionTreeRegressor(max_depth=5) #Modelo a usar
#entrenamiento
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

#Precisión
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f'Error: {mse}')
print(f'R^2 Score: {r2}') 
plt.figure(figsize=(20,10))
plot_tree(modelo, filled=True, feature_names=X.columns, rounded=True)
plt.title('Árbol de Decisión para la prediccion de costos de vivienda')
plt.show()