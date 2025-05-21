import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Evaluación de Créditos",
    page_icon="💰",
    layout="wide"
)

# Cargar el modelo
@st.cache_resource
def load_model():
    with open('modelo_credito.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Intentar cargar el modelo
try:
    model = load_model()
    model_load_success = True
except Exception as e:
    model_load_success = False
    error_message = str(e)

# Título y descripción
st.title("Sistema de Evaluación de Créditos")
st.markdown("""
Esta aplicación evalúa solicitudes de crédito utilizando un modelo de redes neuronales 
(Perceptrón Multicapa) y predice la calificación de riesgo del crédito.
""")

# Definir las columnas y sus tipos para la entrada de datos
numerical_features = ['annual_inc', 'int_rate', 'loan_amnt']
categorical_features = ['loan_status', 'verification_status', 'emp_length', 
                       'home_ownership', 'purpose', 'term']

# Sidebar con información del modelo
st.sidebar.header("Información del Modelo")
st.sidebar.markdown("""
### Modelo: Perceptrón Multicapa (MLP)
- **Solver**: lbfgs
- **Learning Rate**: 0.001
- **Capas Ocultas**: (50, 50, 50)
- **Función de Activación**: tanh
- **Precisión (Test)**: 80.53%
""")

# Funciones auxiliares para la interfaz
def get_input_features():
    st.header("Datos del Solicitante")
    
    col1, col2 = st.columns(2)
    
    with col1:
        annual_inc = st.number_input("Ingresos anuales ($)", min_value=0.0, max_value=1000000.0, value=60000.0, step=1000.0)
        int_rate = st.slider("Tasa de interés (%)", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
        loan_amnt = st.number_input("Monto del crédito ($)", min_value=1000, max_value=50000, value=10000, step=1000)
    
    with col2:
        loan_status = st.selectbox("Estado de la deuda", 
                                  options=["Current", "Fully Paid", "Late", "In Grace Period", "Default"])
        
        verification_status = st.selectbox("Validación de historial crediticio", 
                                         options=["Verified", "Source Verified", "Not Verified"])
        
        emp_length = st.selectbox("Tiempo laborado", 
                                options=["< 1 year", "1 year", "2 years", "3 years", "4 years", 
                                        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        home_ownership = st.selectbox("Tipo de vivienda", 
                                    options=["RENT", "MORTGAGE", "OWN", "OTHER"])
        
        purpose = st.selectbox("Propósito del crédito", 
                             options=["debt_consolidation", "credit_card", "home_improvement", 
                                     "major_purchase", "small_business", "car", "other"])
    
    with col4:
        term = st.selectbox("Duración del crédito", 
                          options=["36 months", "60 months"])

    # Crear un DataFrame con los datos ingresados
    data = {
        'annual_inc': annual_inc,
        'int_rate': int_rate,
        'loan_amnt': loan_amnt,
        'loan_status': loan_status,
        'verification_status': verification_status,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'term': term
    }
    
    features_df = pd.DataFrame(data, index=[0])
    return features_df

def predict_grade(features_df):
    # Aquí deberías aplicar las mismas transformaciones que usaste en el entrenamiento
    # Este es un ejemplo simplificado, deberías adaptarlo según tu pipeline de preprocesamiento
    
    # Predicción
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)
    
    return prediction[0], probability[0]

def display_prediction(prediction, probability):
    st.header("Resultado de la Evaluación")
    
    grade_mapping = {
        'A': 'Riesgo muy bajo',
        'B': 'Riesgo bajo',
        'C': 'Riesgo moderado',
        'D': 'Riesgo medio',
        'E': 'Riesgo medio-alto',
        'F': 'Riesgo alto',
        'G': 'Riesgo muy alto'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Calificación Predicha: {prediction}")
        st.markdown(f"**Interpretación**: {grade_mapping.get(prediction, 'No disponible')}")
    
    with col2:
        # Mostrar probabilidades en forma de gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        grade_labels = model.classes_
        ax.bar(grade_labels, probability)
        ax.set_title('Probabilidad por Calificación')
        ax.set_xlabel('Calificación de Riesgo')
        ax.set_ylabel('Probabilidad')
        st.pyplot(fig)

def main():
    # Verificar si el modelo se cargó correctamente
    if not model_load_success:
        st.error(f"Error al cargar el modelo: {error_message}")
        st.info("Por favor, asegúrate de que el archivo 'modelo_credito.pkl' esté en el mismo directorio que esta aplicación.")
        return

    # Pestañas para organizar el contenido
    tab1, tab2, tab3 = st.tabs(["Predicción", "Datos de Ejemplo", "Acerca del Modelo"])
    
    with tab1:
        st.markdown("### Ingresa los datos del solicitante para evaluar")
        
        # Obtener datos del usuario
        user_input = get_input_features()
        
        # Botón para realizar la predicción
        if st.button("Evaluar Solicitud"):
            with st.spinner('Procesando...'):
                try:
                    # En un escenario real, aquí aplicarías las transformaciones necesarias 
                    # antes de pasar los datos al modelo
                    prediction, probability = predict_grade(user_input)
                    display_prediction(prediction, probability)
                except Exception as e:
                    st.error(f"Error al realizar la predicción: {str(e)}")
                    st.info("Para el funcionamiento correcto del modelo, es necesario que implementes el preprocesamiento adecuado.")
    
    with tab2:
        st.header("Datos de Ejemplo")
        st.markdown("""
        Aquí puedes ver ejemplos de datos para diferentes perfiles de solicitantes:
        """)
        
        ejemplo1 = {
            'annual_inc': 85000,
            'int_rate': 7.5,
            'loan_amnt': 15000,
            'loan_status': 'Current',
            'verification_status': 'Verified',
            'emp_length': '5 years',
            'home_ownership': 'MORTGAGE',
            'purpose': 'debt_consolidation',
            'term': '36 months'
        }
        
        ejemplo2 = {
            'annual_inc': 35000,
            'int_rate': 15.2,
            'loan_amnt': 8000,
            'loan_status': 'Late',
            'verification_status': 'Not Verified',
            'emp_length': '1 year',
            'home_ownership': 'RENT',
            'purpose': 'credit_card',
            'term': '60 months'
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Perfil de Bajo Riesgo")
            st.json(ejemplo1)
            
        with col2:
            st.subheader("Perfil de Alto Riesgo")
            st.json(ejemplo2)
    
    with tab3:
        st.header("Acerca del Modelo")
        st.markdown("""
        ### Perceptrón Multicapa para Evaluación de Crédito
        
        Este modelo utiliza una red neuronal de tipo perceptrón multicapa para clasificar 
        solicitudes de crédito en diferentes grados de riesgo (A-G).
        
        #### Características del modelo:
        - **Arquitectura**: Perceptrón Multicapa (MLP)
        - **Capas ocultas**: 3 capas de 50 neuronas cada una
        - **Función de activación**: Tangente hiperbólica (tanh)
        - **Algoritmo de optimización**: L-BFGS
        - **Precisión en datos de prueba**: 80.53%
        
        #### Variables utilizadas:
        - **loan_status**: Estado de la deuda
        - **annual_inc**: Ingresos anuales
        - **verification_status**: Validación de historial crediticio
        - **emp_length**: Tiempo laborado
        - **home_ownership**: Tipo de vivienda
        - **int_rate**: Tasa de interés
        - **loan_amnt**: Monto del crédito
        - **purpose**: Propósito del crédito
        - **term**: Duración del crédito
        """)
        
        # Mostrar resultados de hiperparámetros
        st.subheader("Resultados de búsqueda de hiperparámetros")
        
        # Recrear tabla de resultados
        resultados_data = {
            'param_solver': ['lbfgs', 'lbfgs', 'lbfgs', 'lbfgs'],
            'param_learning_rate_init': [0.001, 1.000, 1.000, 0.001],
            'param_hidden_layer_sizes': ['(50, 50, 50)', '(20, 20)', '(100, 100)', '10'],
            'param_activation': ['tanh', 'logistic', 'logistic', 'relu'],
            'mean_test_score': [0.805339, 0.739161, 0.729634, 0.708663],
            'std_test_score': [0.006509, 0.015139, 0.007343, 0.007038]
        }
        
        resultados_df = pd.DataFrame(resultados_data)
        st.dataframe(resultados_df)

if __name__ == "__main__":
    main()