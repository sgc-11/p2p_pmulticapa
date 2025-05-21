import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema de Evaluación de Créditos",
    page_icon="💰",
    layout="wide"
)

# ─── Carga de modelo y preprocesador ──────────────────────────────────────────
@st.cache_resource
def load_resources():
    model_path = 'modelo_credito.pkl'
    preproc_path = 'preprocesador_credito.pkl'
    if not os.path.isfile(model_path) or not os.path.isfile(preproc_path):
        raise FileNotFoundError(f"No encontré {model_path!r} o {preproc_path!r} en {os.getcwd()}")
    with open(model_path, 'rb') as f:
        mdl = pickle.load(f)
    with open(preproc_path, 'rb') as f:
        pre = pickle.load(f)
    return mdl, pre

try:
    model, preprocessor = load_resources()
    resources_ok = True
    load_error = ""
except Exception as e:
    resources_ok = False
    load_error = str(e)

# ─── Función helper para extraer categorías conocidas ─────────────────────────
def get_categories(feature_name: str) -> list:
    """
    Busca en el ColumnTransformer el OneHotEncoder (o pipeline con encoder)
    que maneja feature_name y devuelve la lista de categorías que conoce.
    """
    for name, transformer, cols in preprocessor.transformers_:
        if feature_name not in cols:
            continue

        # localizamos el encoder dentro del transformer
        encoder = None
        if hasattr(transformer, 'categories_'):
            encoder = transformer
        elif hasattr(transformer, 'named_steps'):
            # es un Pipeline
            for step in transformer.named_steps.values():
                if hasattr(step, 'categories_'):
                    encoder = step
                    break

        if encoder is None:
            continue

        idx = cols.index(feature_name)
        return list(encoder.categories_[idx])

    # fallback si no lo encontró
    return []

# ─── Datos de ejemplo para pestaña ────────────────────────────────────────────
ejemplo1 = {
    'annual_inc': 85000,
    'int_rate': 7.5,
    'loan_amnt': 15000,
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
    'verification_status': 'Not Verified',
    'emp_length': '1 year',
    'home_ownership': 'RENT',
    'purpose': 'credit_card',
    'term': '60 months'
}

# ─── Interfaz de usuario: inputs dinámicos ────────────────────────────────────
def get_input_features():
    st.header("Datos del Solicitante")

    # obtener listas de categorías reales
    valid_status = get_categories('verification_status')
    valid_length = get_categories('emp_length')
    valid_home   = get_categories('home_ownership')
    valid_purp   = get_categories('purpose')
    valid_term   = get_categories('term')
    dummy_status = get_categories('loan_status')  # para inyectar loan_status

    col1, col2 = st.columns(2)
    with col1:
        annual_inc = st.number_input(
            "Ingresos anuales ($)", min_value=0.0, max_value=1_000_000.0,
            value=60_000.0, step=1_000.0
        )
        int_rate = st.slider(
            "Tasa de interés (%)", min_value=1.0, max_value=30.0,
            value=10.0, step=0.1
        )
        loan_amnt = st.number_input(
            "Monto del crédito ($)", min_value=1_000, max_value=50_000,
            value=10_000, step=1_000
        )
    with col2:
        verification_status = st.selectbox(
            "Validación historial crediticio", options=valid_status
        )
        emp_length = st.selectbox(
            "Tiempo laborado", options=valid_length
        )
    col3, col4 = st.columns(2)
    with col3:
        home_ownership = st.selectbox(
            "Tipo de vivienda", options=valid_home
        )
        purpose = st.selectbox(
            "Propósito del crédito", options=valid_purp
        )
    with col4:
        term = st.selectbox(
            "Duración del crédito", options=valid_term
        )

    data = {
        'annual_inc': annual_inc,
        'int_rate': int_rate,
        'loan_amnt': loan_amnt,
        # inyectamos un loan_status válido aunque no se use
        'loan_status': dummy_status[0] if dummy_status else '',
        'verification_status': verification_status,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'term': term
    }
    return pd.DataFrame(data, index=[0])

# ─── Alineación y predicción ────────────────────────────────────────────────
def predict_and_align(df: pd.DataFrame):
    expected = list(preprocessor.feature_names_in_)
    df2 = df.reindex(columns=expected)
    missing = [c for c in expected if df2[c].isnull().all()]
    if missing:
        st.error(f"Faltan columnas: {missing}")
        return None, None

    Xt = preprocessor.transform(df2)
    X = Xt.toarray() if not isinstance(Xt, np.ndarray) else Xt
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]
    return y_pred, y_proba

# ─── Mostrar resultado ───────────────────────────────────────────────────────
def display_prediction(pred, proba):
    mapping = {
        'A': 'Riesgo muy bajo',
        'B': 'Riesgo bajo',
        'C': 'Riesgo moderado',
        'D': 'Riesgo medio',
        'E': 'Riesgo medio-alto',
        'F': 'Riesgo alto',
        'G': 'Riesgo muy alto'
    }
    st.subheader(f"Calificación: {pred}")
    st.markdown(f"**Interpretación**: {mapping.get(pred,'No disponible')}")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(model.classes_, proba)
    ax.set_ylabel("Probabilidad")
    ax.set_xlabel("Calificación")
    ax.set_title("Distribución de Probabilidades")
    st.pyplot(fig)

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    st.title("Sistema de Evaluación de Créditos")
    if not resources_ok:
        st.error("Error cargando recursos:")
        st.write(load_error)
        return

    tab1, tab2, tab3 = st.tabs(["Predicción","Ejemplos","Acerca"])

    with tab1:
        st.markdown("### Ingresa los datos del solicitante")
        df_in = get_input_features()
        if st.button("Evaluar Solicitud"):
            with st.spinner("Procesando..."):
                pred, proba = predict_and_align(df_in)
                if pred is not None:
                    display_prediction(pred, proba)

    with tab2:
        st.header("Datos de Ejemplo")
        c1, c2 = st.columns(2)
        with c1: st.subheader("Bajo Riesgo"); st.json(ejemplo1)
        with c2: st.subheader("Alto Riesgo");  st.json(ejemplo2)

    with tab3:
        st.header("Acerca del Modelo")
        st.markdown("""
        - **MLP**: 3 capas de 50 neuronas  
        - **Activación**: tanh  
        - **Optimizador**: L-BFGS  
        - **Precisión (test)**: ~80.5%
        """)
        st.subheader("Hiperparámetros evaluados")
        df_res = pd.DataFrame({
            'solver':['lbfgs']*4,
            'learning_rate_init':[0.001,1.0,1.0,0.001],
            'hidden_layer_sizes':['(50,50,50)','(20,20)','(100,100)','(10,)'],
            'activation':['tanh','logistic','logistic','relu'],
            'mean_test_score':[0.805339,0.739161,0.729634,0.708663],
            'std_test_score':[0.006509,0.015139,0.007343,0.007038]
        })
        st.dataframe(df_res)

if __name__ == "__main__":
    main()
