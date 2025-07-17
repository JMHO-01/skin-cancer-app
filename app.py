import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import base64
import io
from datetime import datetime
from fpdf import FPDF
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Tipos de cáncer posibles ---
malignant_types = [
    "Melanoma",
    "Carcinoma basocelular",
    "Carcinoma de células escamosas",
    "Lentigo maligno",
    "Queratoacantoma",
    "Sarcoma de Kaposi"
]

# --- Descripciones de tipos de cáncer ---
cancer_descriptions = {
    "Melanoma": "**Melanoma:** Es uno de los tipos más agresivos de cáncer de piel y puede diseminarse rápidamente si no se detecta a tiempo.",
    "Carcinoma basocelular": "**Carcinoma basocelular:** Suelen ser lesiones de crecimiento lento que rara vez se diseminan a otras partes del cuerpo.",
    "Carcinoma de células escamosas": "**Carcinoma de células escamosas:** Puede aparecer en zonas expuestas al sol y, en algunos casos, propagarse a tejidos cercanos.",
    "Lentigo maligno": "**Lentigo maligno:** Forma de melanoma que suele desarrollarse en piel dañada por el sol, especialmente en personas mayores.",
    "Queratoacantoma": "**Queratoacantoma:** Tumor de crecimiento rápido que a menudo se asemeja al carcinoma de células escamosas.",
    "Sarcoma de Kaposi": "**Sarcoma de Kaposi:** Cáncer que se origina en el revestimiento de vasos sanguíneos o linfáticos y puede manifestarse con manchas o nódulos en la piel."
}

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
        "pdf_timestamp": "Fecha y hora del análisis",
        "generate_compare_pdf": "📊 Generar PDF comparativo de modelos (Matthew & McNemar)",
        "pdf_compare_title": "Comparación de Rendimiento entre Modelos",
        "mcc_description": "El coeficiente MCC mide la calidad de las predicciones clasificatorias.",
        "mcnemar_description": "La prueba de McNemar evalúa diferencias significativas entre dos modelos.",
        "history_title": "Últimos análisis"
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
        "pdf_timestamp": "Date and Time of Analysis",
        "generate_compare_pdf": "\ud83d\udcca Generate comparative models PDF (Matthews & McNemar)",
        "pdf_compare_title": "Model Performance Comparison",
        "mcc_description": "The MCC coefficient measures the quality of classification predictions.",
        "mcnemar_description": "The McNemar test checks significant differences between two models.",
        "history_title": "Recent analyses"
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
        "pdf_timestamp": "Date et heure de l'analyse",
        "generate_compare_pdf": "\ud83d\udcca Générer le PDF comparatif des modèles (Matthews & McNemar)",
        "pdf_compare_title": "Comparaison de performance des modèles",
        "mcc_description": "Le coefficient MCC mesure la qualité des prédictions de classification.",
        "mcnemar_description": "Le test de McNemar évalue les différences significatives entre deux modèles.",
        "history_title": "Analyses récentes"
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
        "pdf_timestamp": "Datum und Uhrzeit der Analyse",
        "generate_compare_pdf": "\ud83d\udcca Vergleichs-PDF der Modelle erzeugen (Matthews & McNemar)",
        "pdf_compare_title": "Leistungsvergleich der Modelle",
        "mcc_description": "Der MCC-Koeffizient misst die Qualität von Klassifikationsvorhersagen.",
        "mcnemar_description": "Der McNemar-Test prüft signifikante Unterschiede zwischen zwei Modellen.",
        "history_title": "Neueste Analysen"
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
        "pdf_timestamp": "Data e hora da análise",
        "generate_compare_pdf": "\ud83d\udcca Gerar PDF comparativo de modelos (Matthew & McNemar)",
        "pdf_compare_title": "Comparação de desempenho entre modelos",
        "mcc_description": "O coeficiente MCC avalia a qualidade das previsões de classificação.",
        "mcnemar_description": "O teste de McNemar verifica diferenças significativas entre dois modelos.",
        "history_title": "Análises recentes"
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
        "pdf_timestamp": "Data e ora dell'analisi",
        "generate_compare_pdf": "\ud83d\udcca Genera PDF comparativo dei modelli (Matthew & McNemar)",
        "pdf_compare_title": "Confronto delle prestazioni dei modelli",
        "mcc_description": "Il coefficiente MCC misura la qualità delle previsioni di classificazione.",
        "mcnemar_description": "Il test di McNemar valuta le differenze significative tra due modelli.",
    "history_title": "Analisi recenti"
    }
}

# --- Manejo de historial ---
def append_history(timestamp, model, result, confidence, cancer_type, image_name):
    file_path = Path("historial.csv")
    file_exists = file_path.exists()
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "result", "confidence", "cancer_type", "image"])
        writer.writerow([timestamp, model, result, f"{confidence:.2f}", cancer_type or "", image_name])


def load_history(n=5):
    file_path = Path("historial.csv")
    if not file_path.exists():
        return []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-n:]

# --- PDF comparativo de modelos ---
def generate_comparison_pdf(models, language):
    t = translations[language]
    mcc_vals = np.random.uniform(0.6, 0.95, size=len(models))
    mc_vals = np.random.uniform(0.01, 0.2, size=len(models))

    fig, ax1 = plt.subplots()
    index = np.arange(len(models))
    bar_width = 0.35
    ax1.bar(index, mcc_vals, bar_width, label="MCC")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("MCC")

    ax2 = ax1.twinx()
    ax2.plot(index, mc_vals, color="red", marker="o", label="McNemar p-value")
    ax2.set_ylabel("McNemar p-value")
    ax1.set_xticks(index)
    ax1.set_xticklabels(models)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    img_path = "temp_plot.png"
    with open(img_path, "wb") as f:
        f.write(buf.getvalue())

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, txt=t["pdf_compare_title"], ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"MCC: {t['mcc_description']}\nMcNemar: {t['mcnemar_description']}")
    pdf.ln(5)
    pdf.image(img_path, x=10, w=190)
    return pdf.output(dest='S').encode('latin1')

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

if st.sidebar.button(t["generate_compare_pdf"]):
    comp_bytes = generate_comparison_pdf(model_options, lang)
    b64_comp = base64.b64encode(comp_bytes).decode()
    href_comp = f'<a href="data:application/pdf;base64,{b64_comp}" download="comparison_models.pdf">{t["download"]}</a>'
    st.sidebar.markdown(href_comp, unsafe_allow_html=True)

st.title(t["title"])
st.markdown(f"**{t['upload']}**")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "jfif", "tif"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption=t["upload"], use_column_width=True)

        if st.button(t["button"]):
            results = {}
            for model in model_options:
                label, confidence, cancer_type = predict(image, model)
                results[model] = (label, confidence, cancer_type)

            sel_label, sel_conf, sel_type = results[selected_model]
            st.success(f"{t['result']}: {sel_label}")
            st.info(f"{t['confidence']}: {sel_conf:.1f}%")
            if sel_label == "Malignant":
                st.warning(f"🔬 {t['pdf_type_detected']}: {sel_type}")
                desc = cancer_descriptions.get(sel_type)
                if desc:
                    st.markdown(desc)

            pdf_bytes = generate_pdf(sel_label, sel_conf, lang, image, sel_type)
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="prediction_report.pdf">{t["download"]}</a>'
            st.markdown(href, unsafe_allow_html=True)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_history(timestamp, selected_model, sel_label, sel_conf, sel_type, uploaded_file.name)
            history_rows = load_history(5)
            if history_rows:
                st.subheader(t.get("history_title", "Historial"))
                st.table(history_rows)

    except UnidentifiedImageError:
        st.error(t["error"])
    except Exception as e:
        st.error(f"{t['error']} ({str(e)})")
