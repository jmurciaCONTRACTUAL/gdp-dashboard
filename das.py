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