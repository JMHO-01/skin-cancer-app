import streamlit as st
from PIL import Image
import numpy as np
import random

# Traducciones
LANGUAGES = {
    'Español': 'es',
    'English': 'en'
}

TRANSLATIONS = {
    'es': {
        'title': 'Simulación de Predicción de Cáncer de Piel',
        'upload_prompt': 'Sube una imagen de una lesión en la piel',
        'submit': 'Analizar Imagen',
        'result': 'Resultado (simulado)',
        'confidence': 'Confianza estimada',
        'malignant': 'Maligno',
        'benign': 'Benigno'
    },
    'en': {
        'title': 'Skin Cancer Prediction (Simulated)',
        'upload_prompt': 'Upload a skin lesion image',
        'submit': 'Analyze Image',
        'result': 'Result (simulated)',
        'confidence': 'Estimated Confidence',
        'malignant': 'Malignant',
        'benign': 'Benign'
    }
}

def main():
    st.set_page_config(page_title="Skin Cancer Demo", layout="centered")

    lang_choice = st.sidebar.selectbox("Idioma / Language", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_choice]
    t = TRANSLATIONS[lang]

    st.title(t['title'])

    uploaded_file = st.file_uploader(t['upload_prompt'], type=["jpg", "jpeg", "png"])

    if uploaded_file and st.button(t['submit']):
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Simulación de predicción
        simulated_class = random.choice([0, 1])
        confidence = round(random.uniform(0.6, 0.95), 2)

        label = t['benign'] if simulated_class == 0 else t['malignant']

        st.success(f"{t['result']}: {label}")
        st.info(f"{t['confidence']}: {confidence*100:.1f}%")

if __name__ == "__main__":
    main()
