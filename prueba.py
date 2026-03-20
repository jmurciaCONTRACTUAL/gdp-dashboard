import numpy as np
import pandas as pd
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64
import sqlite3
import pandas as pd
import os
import seaborn as sns


from google.colab import drive
drive.mount('/content/drive')

os.listdir('/content/drive/MyDrive') #validar lo que se subio en myDrive, un listado de todas las carpetas


archivo = "/content/drive/MyDrive/OFISI/2026_Gestion_OFISI/Contratacion_2026/ModeloSemantico_SeguimientoContratacion/Precontractual2026.xlsx"

df = pd.read_excel(archivo)
df.head()

df.info()
df.describe()
df.isnull().sum()
df = df.dropna()

#####CORRELACIONES DE DATOS######


#df.corr()  # Se comenta la línea problemática

# Para calcular la correlación, necesitas seleccionar solo las columnas numéricas.
# Puedes hacer esto de varias maneras:

# Opción 1: Seleccionar explícitamente las columnas numéricas
numeric_df = df[['BS ITEM', 'AGRUPACION']]
print(numeric_df.corr())

# Opción 2: Seleccionar automáticamente todas las columnas numéricas
numeric_df = df.select_dtypes(include=np.number)
print('Correlación de columnas numéricas:')
print(numeric_df.corr())

sns.histplot(df["BS ITEM"])
plt.show()


sns.set(style="whitegrid")
from sklearn.linear_model import LinearRegression

print("Primeras filas del dataset:")
print(df.head())

df = df.dropna()

## Promedio de cada columna numérica
print("\nPromedios:")
print(df.mean(numeric_only=True))

# Correlación entre variables
correlation = df.corr(numeric_only=True)
print("\nMatriz de correlación:")
print(correlation)

modelo = LinearRegression()
X = numeric_df[['AGRUPACION']] # Independent variable(s)
y = numeric_df['BS ITEM']      # Dependent variable
modelo.fit(X, y) #ejecución de la regresión lineal datos actuales

intercepto = modelo.intercept_ # modelar un modelo que despues de la funcion matematica entregue una predicción del comportamiento de los datos
pendiente = modelo.coef_[0]
print("Intercepto:", intercepto)
print("Pendiente:", pendiente)

# Predicción
agrupacion_nueva = np.array([[55]])# entrenamiento de 55 datos o los primeros
prediccion = modelo.predict(agrupacion_nueva)#modelar una predicción de datos
print("Predicción para AGRUPACION = 55:", prediccion[0])

# Graficar
plt.scatter(X, y, color="blue", label="Datos reales")
# To plot the regression line, predict y-values for all X points or a range of X points
plt.plot(X, modelo.predict(X), color="red", label="Regresión lineal")
plt.xlabel("AGRUPACION")
plt.ylabel("BS ITEM")
plt.legend()
plt.show()

# Histograma de todas las variables numéricas
df.hist(figsize=(10,8))
plt.suptitle("Distribución de Variables")
plt.show()

# Mapa de calor de correlaciones
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()

# Gráfico de dispersión (ejemplo entre dos columnas)
# Cambia "columna1" y "columna2"
if "columna1" in df.columns and "columna6" in df.columns:
    sns.scatterplot(data=df, x="columna1", y="columna6")
    plt.title("Relación entre columna1 y columna6")
    plt.show()


# 8. EXPORTAR RESULTADOS
df.to_csv("datos_limpios.csv", index=False)

print("\nAnálisis finalizado. Archivo limpio guardado como 'datos_limpios.csv'")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for numeric type selection

st.set_page_config(layout="wide") # Optional: set page layout for better dashboard appearance

st.title("Dashboard de Análisis de Datos")

# Previous error explanation:
# The 'HTTP Error 401: Unauthorized' occurred because `pd.read_csv` was attempting
# to access a Google Drive sharing link without proper authorization. Google Drive
# sharing links (like '.../edit?usp=drive_link...') are not direct download links
# that pandas can read directly. They typically require browser interaction or
# specific Google Drive API authentication. Additionally, the original URL
# pointed to an Excel file (.xlsx), not a CSV, meaning `pd.read_csv` would have
# been the wrong function even with authorization.

# Fix: Load the 'datos_limpios.csv' file that was created in the previous cell.
# This assumes the intent is to visualize the data that has already been processed.
try:
    df = pd.read_csv("datos_limpios.csv")
    st.success("Datos cargados correctamente desde 'datos_limpios.csv'")
except FileNotFoundError:
    st.error("Error: 'datos_limpios.csv' no encontrado. Por favor, asegúrate de que el archivo fue exportado en las celdas anteriores.")
    st.stop() # Stop the Streamlit app if data cannot be loaded

st.subheader("Vista de Datos")
st.dataframe(df)

st.subheader("Información del dataset")
st.write(df.describe())

# Select numeric columns for plotting
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if numeric_cols:
    columna_hist = st.selectbox("Selecciona una columna para el histograma", numeric_cols)
    if columna_hist:
        st.subheader("Histograma")
        fig, ax = plt.subplots()
        sns.histplot(df[columna_hist], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlación")
    # Calculate correlation only for numeric columns
    corr = df[numeric_cols].corr()

    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # Gráfico de dispersión (ejemplo entre dos columnas numéricas)
    if len(numeric_cols) >= 2:
        st.subheader("Gráfico de Dispersión")
        col1_scatter = st.selectbox("Selecciona la primera columna para el gráfico de dispersión", numeric_cols, index=0)
        # Ensure index for second selectbox is valid
        col2_scatter_index = 1 if len(numeric_cols) > 1 else 0
        col2_scatter = st.selectbox("Selecciona la segunda columna para el gráfico de dispersión", numeric_cols, index=col2_scatter_index)

        if col1_scatter and col2_scatter and col1_scatter != col2_scatter:
            fig_scatter, ax_scatter = plt.subplots()
            sns.scatterplot(data=df, x=col1_scatter, y=col2_scatter, ax=ax_scatter)
            st.pyplot(fig_scatter)
        else:
            st.info("Selecciona dos columnas diferentes para el gráfico de dispersión.")
    else:
        st.warning("Se necesitan al menos dos columnas numéricas para mostrar un gráfico de dispersión.")
else:
    st.warning("No hay columnas numéricas en el dataset para realizar histogramas o correlaciones.")

# Note: The original 'archivo = st.file_uploader(...)' block was commented out
# and is removed for clarity, as 'datos_limpios.csv' is now the primary input.
# If you wish to allow file uploads, that section would need to be re-integrated
# carefully, possibly making 'datos_limpios.csv' a default or initial state.