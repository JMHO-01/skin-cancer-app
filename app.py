import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import base64
import io
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- Simulated prediction function ---
def simulate_prediction(image, model_name):
    np.random.seed(len(model_name) + len(image.getbands()))
    confidence = np.random.uniform(50, 100)
    label = "Malignant" if confidence > 70 else "Benign"
    return label, confidence

# --- PDF generation function ---
def generate_pdf(result, confidence, language):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    if language == "Español":
        pdf.cell(200, 10, txt="Resultado del Análisis de Cáncer de Piel", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Resultado: {result}", ln=True)
        pdf.cell(200, 10, txt=f"Confianza estimada: {confidence:.2f}%", ln=True)
    else:
        pdf.cell(200, 10, txt="Skin Cancer Analysis Result", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Result: {result}", ln=True)
        pdf.cell(200, 10, txt=f"Estimated confidence: {confidence:.2f}%", ln=True)
    return pdf.output(dest='S').encode('latin1')

# --- Language and interface ---
lang = st.sidebar.selectbox("🌐 Select Language / Selecciona Idioma", ["English", "Español"])
model_options = ["CNN", "Random Forest", "Regresión Lineal"]
selected_model = st.sidebar.selectbox("🧠 Select Model", model_options)

texts = {
    "English": {
        "title": "Skin Cancer Prediction (Simulated)",
        "upload": "Upload a skin lesion image",
        "button": "Analyze Image",
        "result": "Result (simulated)",
        "confidence": "Estimated Confidence",
        "download": "Download PDF",
        "error": "❌ Error processing the image. Please make sure it's a valid file."
    },
    "Español": {
        "title": "Predicción de Cáncer de Piel (Simulado)",
        "upload": "Sube una imagen de lesión en la piel",
        "button": "Analizar Imagen",
        "result": "Resultado (simulado)",
        "confidence": "Confianza Estimada",
        "download": "Descargar PDF",
        "error": "❌ Error al procesar la imagen. Asegúrate de que sea un archivo válido."
    }
}

st.title(texts[lang]["title"])
st.markdown(f"**{texts[lang]['upload']}**")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada" if lang == "Español" else "Loaded image", use_container_width=True)

        if st.button(texts[lang]["button"]):
            results = {}
            for model in model_options:
                label, confidence = simulate_prediction(image, model)
                results[model] = (label, confidence)

            sel_label, sel_conf = results[selected_model]
            st.success(f"{texts[lang]['result']}: {sel_label}")
            st.info(f"{texts[lang]['confidence']}: {sel_conf:.1f}%")

            st.subheader("📊 " + ("Gráfico de Confianza por Modelo" if lang == "Español" else "Confidence Chart by Model"))
            fig, ax = plt.subplots()
            ax.bar(results.keys(), [conf for _, conf in results.values()], color=["green", "blue", "orange"])
            ax.set_ylabel('%')
            ax.set_ylim(0, 100)
            st.pyplot(fig)

            pdf_bytes = generate_pdf(sel_label, sel_conf, lang)
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="prediction_report.pdf">{texts[lang]["download"]}</a>'
            st.markdown(href, unsafe_allow_html=True)

    except UnidentifiedImageError:
        st.error(texts[lang]["error"])
    except Exception as e:
        st.error(f"{texts[lang]['error']} ({str(e)})")

