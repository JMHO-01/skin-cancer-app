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

# --- PDF generation function with image and recommendations ---
def generate_pdf(result, confidence, language, pil_image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    # Título
    if language == "Español":
        pdf.cell(0, 10, "Resultado del Análisis de Cáncer de Piel", ln=True, align='C')
    elif language == "Français":
        pdf.cell(0, 10, "Résultat de l'analyse du cancer de la peau", ln=True, align='C')
    else:
        pdf.cell(0, 10, "Skin Cancer Analysis Result", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    # Resultado
    pdf.cell(0, 10, f"{'Resultado' if language == 'Español' else 'Result'}: {result}", ln=True)
    pdf.cell(0, 10, f"{'Confianza estimada' if language == 'Español' else 'Estimated confidence'}: {confidence:.2f}%", ln=True)
    pdf.ln(10)

    # Descripción
    description = {
        "Español": "Este resultado ha sido generado a través de un modelo de predicción simulado. Se recomienda no tomar decisiones médicas basadas únicamente en esta evaluación.",
        "Français": "Ce résultat a été généré à l'aide d'un modèle simulé. Il est déconseillé de prendre des décisions médicales uniquement sur cette base.",
        "English": "This result has been generated using a simulated prediction model. Please do not make medical decisions based solely on this result."
    }
    pdf.multi_cell(0, 10, description.get(language, description["English"]))
    pdf.ln(5)

    # Recomendación
    if result == "Malignant":
        reco = {
            "Español": "🔴 Recomendación: Acude a un dermatólogo lo antes posible para una evaluación profesional.",
            "Français": "🔴 Recommandation : Consultez un dermatologue dès que possible pour un avis professionnel.",
            "English": "🔴 Recommendation: See a dermatologist as soon as possible for a professional evaluation."
        }
    else:
        reco = {
            "Español": "🟢 Recomendación: Continúa monitoreando la zona y consulta con un especialista si observas cambios.",
            "Français": "🟢 Recommandation : Continuez à surveiller la zone et consultez un spécialiste si vous remarquez des changements.",
            "English": "🟢 Recommendation: Keep monitoring the area and consult a specialist if you notice changes."
        }
    pdf.multi_cell(0, 10, reco.get(language, reco["English"]))
    pdf.ln(5)

    # Agregar imagen evaluada
    image_path = "/tmp/evaluated_image.jpg"
    pil_image.save(image_path)
    pdf.image(image_path, x=40, w=130)

    return pdf.output(dest='S').encode('latin1')

# --- Idiomas disponibles ---
lang = st.sidebar.selectbox("🌐 Select Language / Selecciona Idioma", ["English", "Español", "Français"])
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
    },
    "Français": {
        "title": "Prédiction du cancer de la peau (simulée)",
        "upload": "Téléversez une image de lésion cutanée",
        "button": "Analyser l'image",
        "result": "Résultat (simulé)",
        "confidence": "Confiance estimée",
        "download": "Télécharger le PDF",
        "error": "❌ Erreur lors du traitement de l'image. Assurez-vous que le fichier est valide."
    }
}

st.title(texts[lang]["title"])
st.markdown(f"**{texts[lang]['upload']}**")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "jfif", "tif"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada" if lang == "Español" else "Image chargée" if lang == "Français" else "Loaded image", use_column_width=True)

        if st.button(texts[lang]["button"]):
            results = {}
            for model in model_options:
                label, confidence = simulate_prediction(image, model)
                results[model] = (label, confidence)

            sel_label, sel_conf = results[selected_model]
            st.success(f"{texts[lang]['result']}: {sel_label}")
            st.info(f"{texts[lang]['confidence']}: {sel_conf:.1f}%")

            st.subheader("📊 " + (
                "Gráfico de Confianza por Modelo" if lang == "Español"
                else "Graphique de confiance par modèle" if lang == "Français"
                else "Confidence Chart by Model"
            ))

            fig, ax = plt.subplots()
            ax.bar(results.keys(), [conf for _, conf in results.values()], color=["green", "blue", "orange"])
            ax.set_ylabel('%')
            ax.set_ylim(0, 100)
            st.pyplot(fig)

            pdf_bytes = generate_pdf(sel_label, sel_conf, lang, image)
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="prediction_report.pdf">{texts[lang]["download"]}</a>'
            st.markdown(href, unsafe_allow_html=True)

    except UnidentifiedImageError:
        st.error(texts[lang]["error"])
    except Exception as e:
        st.error(f"{texts[lang]['error']} ({str(e)})")
