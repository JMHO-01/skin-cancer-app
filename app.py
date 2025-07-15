import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# --- Cargar el modelo H5 una sola vez ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_skincancer2.h5")
    return model

model = load_model()

# --- Traducciones ---
LANGUAGES = {
    'Español': 'es',
    'English': 'en'
}

TRANSLATIONS = {
    'es': {
        'title': 'Predicción de Cáncer de Piel',
        'upload_prompt': 'Sube una imagen de una lesión en la piel',
        'submit': 'Analizar Imagen',
        'result': 'Resultado',
        'confidence': 'Confianza',
        'malignant': 'Maligno',
        'benign': 'Benigno'
    },
    'en': {
        'title': 'Skin Cancer Prediction',
        'upload_prompt': 'Upload a skin lesion image',
        'submit': 'Analyze Image',
        'result': 'Result',
        'confidence': 'Confidence',
        'malignant': 'Malignant',
        'benign': 'Benign'
    }
}

# --- Preprocesamiento de imagen ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((64, 64))  # Tamaño usado en tu notebook
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.set_page_config(page_title="Skin Cancer App", layout="centered")

    lang_choice = st.sidebar.selectbox("Idioma / Language", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_choice]
    t = TRANSLATIONS[lang]

    st.title(t['title'])

    uploaded_file = st.file_uploader(t['upload_prompt'], type=["jpg", "jpeg", "png"])

    if uploaded_file and st.button(t['submit']):
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Imagen cargada", use_column_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        label = t['benign'] if predicted_class == 0 else t['malignant']

        st.success(f"{t['result']}: {label}")
        st.info(f"{t['confidence']}: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
