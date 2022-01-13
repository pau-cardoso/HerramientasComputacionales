import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

file = './data.csv'

##############################################################
# Estadísticos
##############################################################

df = pd.read_csv(file)
print("\nNúmero de variables:", df.shape[1])     #tupla con número de renglones y número de columnas
print("Número de registros:", df.shape[0])     #tupla con número de renglones y número de columnas

print("\n****** Nombre de columnas ******")   #lista con nombres de las columnas. Se puede cambiar el nombre de las columnas al asignar otro valor a este atributo
for column in df.columns:
	print("-", column)

print("\n******** Tipos de datos ********\n", df.dtypes)    #tipos de datos de cada columna


# quita los renglones (axis=0) que contienen cualquier (how='any', 'all') columna vacía, inplace significa que modifica el dataframe df
df.dropna(axis = 0, how = 'any', inplace = True)


# Estadísticos de la columna 'danceability'
print("\n***** Estadísticos de la columna 'danceability' *****")
print("\nValores únicos:", df["danceability"].unique()) # Valores únicos
print("\nValor máximo:", df["danceability"].max()) # Valor máximo
print("Valor mínimo:", df["danceability"].min()) # Valor mínimo
print("\nPromedio:", df["danceability"].mean()) # Promedio
print("Mediana:", df["danceability"].median()) # Mediana
print("Desviación estándar:", df["danceability"].std()) # Desviación estándar

# Estadísticos de la columna 'key'
print("\n******* Estadísticos de la columna 'key' *******")
print("\nValores únicos:", df["key"].unique()) # Valores únicos
print("\nValor máximo:", df["key"].max()) # Valor máximo
print("Valor mínimo:", df["key"].min()) # Valor mínimo
print("\nPromedio:", df["key"].mean()) # Promedio
print("Mediana:", df["key"].median()) # Mediana
print("Desviación estándar:", df["key"].std()) # Desviación estándar


##############################################################
# Gráficos
##############################################################

# ******* Gráficos de columna 'danceability' *******
# Histograma
df.hist(column="danceability", grid=False, color = "coral")
plt.show()

# Diagrama de cajas y bigotes
df.boxplot(column=["danceability"], color = "blue", showmeans=True )
plt.show()

# ******* Gráficos de columna 'key' *******
# Histograma
df.hist(column="key", grid=False, color = "#66CDAA")
plt.show()

# Diagrama de cajsa y bigotes
df.boxplot(column=["key"], color = "blue", showmeans=True )
plt.show()


# Mapa de calor
plt.figure(figsize=(15, 5))
sns.heatmap(df.corr(), annot=True, cmap="summer");
plt.show()


##############################################################
#  k means
##############################################################
print("\n********** K-means **********")

test = df[["valence","energy"]]
test = test.dropna(axis = 0, how = 'any')

kmeans = KMeans(n_clusters=3).fit(test)
centroids = kmeans.cluster_centers_
print("\nCentros\n", centroids)

# Predicciones (cuál es la clase) de acuerdo a los centros calculados
cla = kmeans.predict(test)                   # obtiene las clases de los datos iniciales

plt.scatter(df["valence"],df["energy"],c=cla)
for i in range(len(centroids)):
    plt.scatter(centroids[i][0],centroids[i][1],marker="*",c="red")
plt.show()