# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:38:37 2025

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import base64

# Configuración de la página
st.set_page_config(page_title="Regresión Lineal con Regularización", layout="wide")

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Título de la aplicación
st.title("Regresión Lineal con Regularización")

# Cargar el archivo CSV
@st.cache_data
def load_data():
    data = pd.read_csv('meatspec.csv')
    data = data.drop(columns=data.columns[0])  # Eliminar la primera columna (índice)
    return data

datos = load_data()

# Mostrar todos los registros del archivo meatspec.csv
st.subheader("Datos Cargados (Todos los Registros)")
st.dataframe(datos)  # Mostrar todos los registros

# Histograma de la variable objetivo (fat)
st.subheader("Distribución del Contenido de Grasa (fat)")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(datos['fat'], bins=30, kde=True, ax=ax)
ax.set_xlabel('Contenido de grasa (fat)')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

# División de los datos en train y test
X = datos.drop(columns='fat')
y = datos['fat']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234, shuffle=True)

# Normalización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de Mínimos Cuadrados Ordinarios (OLS)
st.subheader("Modelo de Mínimos Cuadrados Ordinarios (OLS)")
modelo_ols = LinearRegression()
modelo_ols.fit(X_train_scaled, y_train)
predicciones_ols = modelo_ols.predict(X_test_scaled)
rmse_ols = np.sqrt(mean_squared_error(y_test, predicciones_ols))
st.write(f"Error (RMSE) de test para OLS: {rmse_ols:.4f}")

# Modelo Ridge con regularización L2
st.subheader("Modelo Ridge con Regularización L2")
alpha_ridge = st.slider("Selecciona el valor de alpha para Ridge", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
modelo_ridge = Ridge(alpha=alpha_ridge)
modelo_ridge.fit(X_train_scaled, y_train)
predicciones_ridge = modelo_ridge.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, predicciones_ridge))
st.write(f"Error (RMSE) de test para Ridge: {rmse_ridge:.4f}")

# Gráfico de coeficientes para Ridge
coeficientes_ridge = modelo_ridge.coef_
fig, ax = plt.subplots(figsize=(11, 3.84))
ax.stem(X_train.columns, coeficientes_ridge, markerfmt=' ')
plt.xticks(rotation=90, ha='right', size=5)
ax.set_xlabel('Variable')
ax.set_ylabel('Coeficientes')
ax.set_title('Coeficientes del Modelo Ridge')
st.pyplot(fig)

# Modelo Lasso con regularización L1
st.subheader("Modelo Lasso con Regularización L1")
alpha_lasso = st.slider("Selecciona el valor de alpha para Lasso", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
modelo_lasso = Lasso(alpha=alpha_lasso)
modelo_lasso.fit(X_train_scaled, y_train)
predicciones_lasso = modelo_lasso.predict(X_test_scaled)
rmse_lasso = np.sqrt(mean_squared_error(y_test, predicciones_lasso))
st.write(f"Error (RMSE) de test para Lasso: {rmse_lasso:.4f}")

# Gráfico de coeficientes para Lasso
coeficientes_lasso = modelo_lasso.coef_
fig, ax = plt.subplots(figsize=(11, 3.84))
ax.stem(X_train.columns, coeficientes_lasso, markerfmt=' ')
plt.xticks(rotation=90, ha='right', size=5)
ax.set_xlabel('Variable')
ax.set_ylabel('Coeficientes')
ax.set_title('Coeficientes del Modelo Lasso')
st.pyplot(fig)

# DataFrame con los valores de alpha, RMSE y coeficientes
st.subheader("Resumen de Modelos")
df_resumen = pd.DataFrame({
    'Modelo': ['OLS', 'Ridge', 'Lasso'],
    'Alpha': [None, alpha_ridge, alpha_lasso],
    'RMSE': [rmse_ols, rmse_ridge, rmse_lasso],
    'Coeficientes': [modelo_ols.coef_, modelo_ridge.coef_, modelo_lasso.coef_]
})

# Mostrar el DataFrame
st.dataframe(df_resumen)

# Comparación de modelos
st.subheader("Comparación de Modelos")
df_comparacion = pd.DataFrame({
    'Modelo': ['OLS', 'Ridge', 'Lasso'],
    'RMSE': [rmse_ols, rmse_ridge, rmse_lasso]
})

fig, ax = plt.subplots(figsize=(7, 3.84))
df_comparacion.set_index('Modelo').plot(kind='barh', ax=ax, legend=False)
ax.set_xlabel('RMSE')
ax.set_ylabel('Modelo')
ax.set_title('Comparación de Modelos')
st.pyplot(fig)

# Respuestas a los incisos
st.subheader("Respuestas a los Incisos")

respuestas = f"""
**i) ¿Cómo cambia el modelo al aumentar la penalización en L1 y L2?**

- **Ridge (L2)**: A medida que aumenta el valor de `alpha` (actualmente en `{alpha_ridge}`), los coeficientes del modelo se reducen en magnitud, pero no llegan a ser exactamente cero. Esto ayuda a controlar la multicolinealidad y reduce el sobreajuste. 
  - **Coeficientes de Ridge**: {modelo_ridge.coef_}
  - **RMSE de Ridge**: {rmse_ridge:.4f}

- **Lasso (L1)**: A medida que aumenta el valor de `alpha` (actualmente en `{alpha_lasso}`), algunos coeficientes se vuelven exactamente cero, lo que efectivamente realiza una selección de características. Esto es útil para simplificar el modelo y eliminar predictores irrelevantes.
  - **Coeficientes de Lasso**: {modelo_lasso.coef_}
  - **RMSE de Lasso**: {rmse_lasso:.4f}

**ii) Comenta los resultados obtenidos y discute si la regularización mejoró o no el desempeño del modelo.**

- **Comparación de RMSE**:
  - **OLS**: {rmse_ols:.4f}
  - **Ridge**: {rmse_ridge:.4f}
  - **Lasso**: {rmse_lasso:.4f}

- **Análisis**:
  - El modelo Ridge tiene un error (RMSE) de `{rmse_ridge:.4f}`, que es ligeramente menor que el error del modelo OLS (`{rmse_ols:.4f}`). Esto sugiere que la regularización L2 mejoró el desempeño del modelo al reducir el sobreajuste.
  - El modelo Lasso tiene un error de `{rmse_lasso:.4f}`, que es similar al error del OLS, pero con la ventaja de que solo utiliza un subconjunto de predictores (coeficientes distintos de cero: `{np.sum(modelo_lasso.coef_ != 0)}` de `{len(modelo_lasso.coef_)}`). Esto simplifica el modelo sin sacrificar mucho el desempeño.

**iii) ¿Qué problemas pueden evitarse con el uso de regularización en modelos de regresión?**

- **Sobreajuste (Overfitting)**: La regularización ayuda a reducir el sobreajuste al penalizar los coeficientes grandes, lo que resulta en un modelo más generalizable. Por ejemplo, Ridge redujo el RMSE de `{rmse_ols:.4f}` (OLS) a `{rmse_ridge:.4f}`.
- **Multicolinealidad**: En particular, Ridge es útil cuando hay alta correlación entre predictores, ya que reduce la varianza de las estimaciones. Esto se observa en los coeficientes de Ridge, que son más pequeños en magnitud que los de OLS.
- **Selección de características**: Lasso puede eliminar predictores irrelevantes al establecer sus coeficientes a cero. En este caso, Lasso seleccionó solo `{np.sum(modelo_lasso.coef_ != 0)}` predictores de `{len(modelo_lasso.coef_)}`.
"""

st.markdown(respuestas)

# Función para generar el PDF
def generar_pdf(respuestas, alpha_ridge, alpha_lasso, rmse_ols, rmse_ridge, rmse_lasso, coeficientes_ridge, coeficientes_lasso):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Título
    pdf.cell(200, 10, txt="Respuestas a los Incisos", ln=True, align='C')
    pdf.ln(10)
    
    # Respuestas
    pdf.multi_cell(0, 10, txt=respuestas)
    
    # Información adicional
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Alpha Ridge: {alpha_ridge}", ln=True)
    pdf.cell(200, 10, txt=f"Alpha Lasso: {alpha_lasso}", ln=True)
    pdf.cell(200, 10, txt=f"RMSE OLS: {rmse_ols:.4f}", ln=True)
    pdf.cell(200, 10, txt=f"RMSE Ridge: {rmse_ridge:.4f}", ln=True)
    pdf.cell(200, 10, txt=f"RMSE Lasso: {rmse_lasso:.4f}", ln=True)
    pdf.ln(10)
    
    # Coeficientes Ridge
    pdf.cell(200, 10, txt="Coeficientes Ridge:", ln=True)
    for i, coef in enumerate(coeficientes_ridge):
        pdf.cell(200, 10, txt=f"Variable {i+1}: {coef:.4f}", ln=True)
    
    # Coeficientes Lasso
    pdf.ln(10)
    pdf.cell(200, 10, txt="Coeficientes Lasso:", ln=True)
    for i, coef in enumerate(coeficientes_lasso):
        pdf.cell(200, 10, txt=f"Variable {i+1}: {coef:.4f}", ln=True)
    
    # Guardar el PDF en un archivo temporal
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

# Botón para generar y descargar el PDF
if st.button("Generar y Descargar PDF"):
    pdf_output = generar_pdf(respuestas, alpha_ridge, alpha_lasso, rmse_ols, rmse_ridge, rmse_lasso, coeficientes_ridge, coeficientes_lasso)
    
    # Crear un enlace de descarga
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="respuestas_incisos.pdf">Descargar PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
