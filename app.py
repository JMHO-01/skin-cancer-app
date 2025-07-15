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
        'title': 'Simulación de Predicción con Múltiples Modelos',
        'upload_prompt': 'Sube una imagen de una lesión en la piel',
        'submit': 'Analizar Imagen',
        'results': 'Resultados por Modelo (Simulados)',
        'final_diagnosis': 'Diagnóstico Final',
        'malignant': 'Maligno',
        'benign': 'Benigno'
    },
    'en': {
        'title': 'Simulated Prediction with Multiple Models',
        'upload_prompt': 'Upload a skin lesion image',
        'submit': 'Analyze Image',
        'results': 'Model-wise Results (Simulated)',
        'final_diagnosis': 'Final Diagnosis',
        'malignant': 'Malignant',
        'benign': 'Benign'
    }
}

MODELS = [
    {'name': 'CNN', 'range': (0.75, 0.95)},
    {'name': 'Random Forest', 'range': (0.60, 0.90)},
    {'name': 'Regresión Logística', 'range': (0.55, 0.85)}
]

def simulate_model_result(prob_range):
    """Simula la predicción de un modelo dado un rango de confianza"""
    prediction = random.choice([0, 1])  # 0: Benigno, 1: Maligno
    confidence = round(random.uniform(*prob_range), 2)
    return prediction, confidence

def main():
    st.set_page_config(page_title="Skin Cancer Simulation", layout="centered")

    lang_choice = st.sidebar.selectbox("Idioma / Language", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_choice]
    t = TRANSLATIONS[lang]

    st.title(t['title'])

    uploaded_file = st.file_uploader(t['upload_prompt'], type=["jpg", "jpeg", "png"])

    if uploaded_file and st.button(t['submit']):
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Imagen cargada", use_container_width=True)

        st.subheader(t['results'])
        results = []
        malignant_votes = 0

        for model in MODELS:
            prediction, confidence = simulate_model_result(model['range'])
            label = t['malignant'] if prediction == 1 else t['benign']
            results.append((model['name'], label, confidence))

            if prediction == 1:
                malignant_votes += 1

            st.write(f"**{model['name']}** → {label} ({confidence * 100:.1f}%)")

        # Diagnóstico final
        final = t['malignant'] if malignant_votes >= 2 else t['benign']
        st.markdown(f"### 🧪 {t['final_diagnosis']}: **{final}**")

if __name__ == "__main__":
    main()
