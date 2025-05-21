# Sistema de Evaluación de Créditos

Este repositorio contiene una aplicación web interactiva construida con Streamlit que evalúa solicitudes de crédito usando un modelo de Perceptrón Multicapa (MLP) previamente entrenado y serializado. La app carga un preprocesador y un modelo desde archivos pickle, recibe datos del solicitante, aplica el pipeline de transformación y muestra la predicción junto con un gráfico de probabilidades.

---

## 📁 Estructura del repositorio
├── app.py
├── modelo_credito.pkl
├── preprocesador_credito.pkl
├── requirements.txt
└── README.md


- **app.py**  
  Código principal de la aplicación Streamlit.  
- **modelo_credito.pkl**  
  Archivo pickle con el modelo MLP serializado.  
- **preprocesador_credito.pkl**  
  Archivo pickle con el `ColumnTransformer` / pipeline de preprocesamiento.  
- **requirements.txt**  
  Lista de dependencias de Python necesarias.  
- **README.md**  
  Guía de instalación y uso (este documento).

---

## 🛠️ Requisitos Previos

- Python 3.8 o superior  
- Git (para clonar el repositorio)  
- Opcionalmente, [virtualenv](https://virtualenv.pypa.io/) o [conda](https://docs.conda.io/) para crear un entorno aislado

---

## 🚀 Instalación

1. **Clona este repositorio**  
   ```bash
   git clone https://github.com/tu-usuario/credito-streamlit.git
   cd credito-streamlit

2. **Crea en entorno virtual**
```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\activate       # Windows
```
3. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```bash
streamlit run app.py
```
