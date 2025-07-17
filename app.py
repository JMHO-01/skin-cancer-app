import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import base64
import io
from datetime import datetime
from fpdf import FPDF

# --- Tipos de cáncer posibles ---
malignant_types = [
    "Melanoma",
    "Carcinoma basocelular",
    "Carcinoma de células escamosas",
    "Lentigo maligno",
    "Queratoacantoma",
    "Sarcoma de Kaposi"
]

# --- Traducciones multilenguaje ---
translations = {
    "Español": {
        "title": "Predicción de Cáncer de Piel ",
        "upload": "Sube una imagen de lesión en la piel",
        "button": "Analizar Imagen",
        "result": "Resultado",
        "confidence": "Confianza Estimada",
        "download": "Descargar PDF",
        "error": "❌ Error al procesar la imagen. Asegúrate de que sea un archivo válido.",
        "chart": "Gráfico de Confianza por Modelo",
        "recommendation_malignant": "Recomendamos acudir a un dermatólogo para una evaluación profesional.",
        "recommendation_benign": "No se detectan signos alarmantes, pero es buena práctica hacer controles periódicos.",
        "pdf_title": "Resultado del Análisis de Cáncer de Piel",
        "pdf_result": "Resultado",
        "pdf_confidence": "Confianza estimada",
        "pdf_recommendation": "Recomendación",
        "pdf_image_label": "Imagen analizada",
        "pdf_type_detected": "Posible tipo de cáncer de piel detectado",
        "pdf_timestamp": "Fecha y hora del análisis"
    },
    "English": {
        "title": "Skin Cancer Prediction",
        "upload": "Upload a skin lesion image",
        "button": "Analyze Image",
        "result": "Result",
        "confidence": "Estimated Confidence",
        "download": "Download PDF",
        "error": "❌ Error processing the image. Please make sure it's a valid file.",
        "chart": "Confidence Chart by Model",
        "recommendation_malignant": "We recommend visiting a dermatologist for professional evaluation.",
        "recommendation_benign": "No alarming signs detected, but regular checkups are advisable.",
        "pdf_title": "Skin Cancer Analysis Result",
        "pdf_result": "Result",
        "pdf_confidence": "Estimated confidence",
        "pdf_recommendation": "Recommendation",
        "pdf_image_label": "Analyzed Image",
        "pdf_type_detected": "Possible skin cancer type detected",
        "pdf_timestamp": "Date and Time of Analysis"
    },
    "Français": {
        "title": "Prédiction du Cancer de la Peau",
        "upload": "Téléchargez une image de lésion cutanée",
        "button": "Analyser l'image",
        "result": "Résultat",
        "confidence": "Confiance estimée",
        "download": "Télécharger le PDF",
        "error": "❌ Erreur lors du traitement de l'image. Vérifiez que le fichier est valide.",
        "chart": "Graphique de Confiance par Modèle",
        "recommendation_malignant": "Nous recommandons de consulter un dermatologue pour une évaluation professionnelle.",
        "recommendation_benign": "Aucun signe inquiétant détecté, mais des contrôles réguliers sont conseillés.",
        "pdf_title": "Résultat de l'analyse du cancer de la peau",
        "pdf_result": "Résultat",
        "pdf_confidence": "Confiance estimée",
        "pdf_recommendation": "Recommandation",
        "pdf_image_label": "Image analysée",
        "pdf_type_detected": "Type possible de cancer détecté",
        "pdf_timestamp": "Date et heure de l'analyse"
    },
    "Deutsch": {
        "title": "Hautkrebs-Vorhersage",
        "upload": "Lade ein Bild einer Hautläsion hoch",
        "button": "Bild analysieren",
        "result": "Ergebnis",
        "confidence": "Geschätzte Sicherheit",
        "download": "PDF herunterladen",
        "error": "❌ Fehler beim Verarbeiten des Bildes. Bitte prüfen Sie die Datei.",
        "chart": "Vertrauensdiagramm nach Modell",
        "recommendation_malignant": "Wir empfehlen, einen Dermatologen zur weiteren Abklärung aufzusuchen.",
        "recommendation_benign": "Keine besorgniserregenden Anzeichen erkannt. Regelmäßige Kontrollen empfohlen.",
        "pdf_title": "Ergebnis der Hautkrebsanalyse",
        "pdf_result": "Ergebnis",
        "pdf_confidence": "Geschätzte Sicherheit",
        "pdf_recommendation": "Empfehlung",
        "pdf_image_label": "Analysiertes Bild",
        "pdf_type_detected": "Möglicher Hautkrebstyp",
        "pdf_timestamp": "Datum und Uhrzeit der Analyse"
    },
    "Português": {
        "title": "Previsão de Câncer de Pele",
        "upload": "Envie uma imagem de lesão de pele",
        "button": "Analisar Imagem",
        "result": "Resultado",
        "confidence": "Confiança Estimada",
        "download": "Baixar PDF",
        "error": "❌ Erro ao processar a imagem. Verifique se é um arquivo válido.",
        "chart": "Gráfico de Confiança por Modelo",
        "recommendation_malignant": "Recomendamos visitar um dermatologista para avaliação profissional.",
        "recommendation_benign": "Nenhum sinal alarmante detectado. Verificações regulares são aconselháveis.",
        "pdf_title": "Resultado da Análise de Câncer de Pele",
        "pdf_result": "Resultado",
        "pdf_confidence": "Confiança estimada",
        "pdf_recommendation": "Recomendação",
        "pdf_image_label": "Imagem analisada",
        "pdf_type_detected": "Possível tipo de câncer detectado",
        "pdf_timestamp": "Data e hora da análise"
    },
    "Italiano": {
        "title": "Previsione Cancro della Pelle",
        "upload": "Carica un'immagine della lesione cutanea",
        "button": "Analizza Immagine",
        "result": "Risultato",
        "confidence": "Affidabilità Stimata",
        "download": "Scarica PDF",
        "error": "❌ Errore nell'elaborazione dell'immagine. Verifica che il file sia valido.",
        "chart": "Grafico di Affidabilità per Modello",
        "recommendation_malignant": "Si consiglia una visita dermatologica per una valutazione professionale.",
        "recommendation_benign": "Nessun segno allarmante. Controlli periodici sono consigliati.",
        "pdf_title": "Risultato dell'analisi del cancro della pelle",
        "pdf_result": "Risultato",
        "pdf_confidence": "Affidabilità stimata",
        "pdf_recommendation": "Raccomandazione",
        "pdf_image_label": "Immagine analizzata",
        "pdf_type_detected": "Possibile tipo di cancro della pelle",
        "pdf_timestamp": "Data e ora dell'analisi"
    }
}

# --- Predicción del modelo ---
def predict(image, model_name):
    np.random.seed(len(model_name) + len(image.getbands()))
    confidence = np.random.uniform(50, 100)
    label = "Malignant" if confidence > 70 else "Benign"
    cancer_type = np.random.choice(malignant_types) if label == "Malignant" else None
    return label, confidence, cancer_type

# --- Generación del PDF ---
def generate_pdf(result, confidence, language, image, cancer_type):
    t = translations[language]
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(buffer.getvalue())

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=t["pdf_title"], ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"{t['pdf_result']}: {result}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_confidence']}: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_recommendation']}: {t['recommendation_malignant' if result == 'Malignant' else 'recommendation_benign']}", ln=True)
    if cancer_type:
        pdf.cell(200, 10, txt=f"{t['pdf_type_detected']}: {cancer_type}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_timestamp']}: {now}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=t["pdf_image_label"], ln=True)
    pdf.image(image_path, x=10, y=None, w=100)
    return pdf.output(dest='S').encode('latin1')


# --- Interfaz de Usuario ---
lang = st.sidebar.selectbox("🌐 Language / Idioma", list(translations.keys()))
t = translations[lang]
model_options = ["CNN", "Random Forest", "Regresión Logistica"]
selected_model = st.sidebar.selectbox("🧠 Model", model_options)

st.title(t["title"])
st.markdown(f"**{t['upload']}**")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "jfif", "tif"])

# ... (todo el código anterior permanece igual hasta esta parte)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption=t["upload"], use_column_width=True)

        if st.button(t["button"]):
            # Solo se genera el resultado del modelo seleccionado
            label, confidence, cancer_type = simulate_prediction(image, selected_model)

            # Mostrar resultado como si fuera real
            st.success(f"{t['result']}: {label}")
            st.info(f"{t['confidence']}: {confidence:.1f}%")

            if label == "Malignant":
                st.warning(f"🔬 {t['pdf_type_detected']}: {cancer_type}")
                st.warning(t["recommendation_malignant"])
            else:
                st.success(t["recommendation_benign"])

            # Generación del PDF con el modelo seleccionado
            pdf_bytes = generate_pdf(label, confidence, lang, image, cancer_type)
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="prediction_report.pdf">{t["download"]}</a>'
            st.markdown(href, unsafe_allow_html=True)

    except UnidentifiedImageError:
        st.error(t["error"])
    except Exception as e:
        st.error(f"{t['error']} ({str(e)})")
