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
        "mcc_chart_title": "Coeficiente de Mathews (MCC) por modelo",
        "mcnemar_chart_title": "Prueba de McNemar – Comparación entre modelos",
        "mcc_description": "El coeficiente MCC mide la calidad de las predicciones clasificatorias. Random Forest obtuvo el valor más alto, indicando mejor rendimiento.",
        "mcnemar_description": "La prueba de McNemar evalúa diferencias significativas entre dos modelos. Los valores muestran ventajas significativas a favor de Random Forest.",
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
        "mcc_chart_title": "Matthews Correlation Coefficient (MCC) by model",
        "mcnemar_chart_title": "McNemar Test \u2013 Model comparisons",
        "mcc_description": "The MCC coefficient measures the quality of classification predictions. Random Forest achieved the highest score, indicating superior performance.",
        "mcnemar_description": "The McNemar test checks significant differences between two models. The results favor Random Forest over the other models.",
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
        "mcc_chart_title": "Coefficient de Matthews (MCC) par modèle",
        "mcnemar_chart_title": "Test de McNemar \u2013 Comparaison des modèles",
        "mcc_description": "Le coefficient MCC mesure la qualité des prédictions de classification. Random Forest obtient la meilleure valeur, signe d'une performance supérieure.",
        "mcnemar_description": "Le test de McNemar évalue les différences significatives entre deux modèles. Les r\u00e9sultats favorisent Random Forest par rapport aux autres.",
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
        "mcc_chart_title": "Matthews-Korrelation (MCC) nach Modell",
        "mcnemar_chart_title": "McNemar-Test \u2013 Modellvergleiche",
        "mcc_description": "Der MCC-Koeffizient misst die Qualität von Klassifikationsvorhersagen. Random Forest erzielt den höchsten Wert und liefert die beste Leistung.",
        "mcnemar_description": "Der McNemar-Test pr\u00fcft signifikante Unterschiede zwischen zwei Modellen. Die Ergebnisse sprechen deutlich f\u00fcr Random Forest.",
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
        "mcc_chart_title": "Coeficiente de Matthews (MCC) por modelo",
        "mcnemar_chart_title": "Teste de McNemar \u2013 Comparação entre modelos",
        "mcc_description": "O coeficiente MCC avalia a qualidade das previsões de classificação. O Random Forest apresentou o valor mais alto, indicando melhor desempenho.",
        "mcnemar_description": "O teste de McNemar verifica diferenças significativas entre dois modelos. Os resultados favorecem o Random Forest em relacao aos demais.",
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
        "mcc_chart_title": "Coefficiente di Matthews (MCC) per modello",
        "mcnemar_chart_title": "Test di McNemar \u2013 Confronto tra modelli",
        "mcc_description": "Il coefficiente MCC misura la qualit\u00e0 delle previsioni di classificazione. Random Forest ottiene il valore pi\u00f9 alto, indicando prestazioni superiori.",
        "mcnemar_description": "Il test di McNemar valuta le differenze significative tra due modelli. I risultati evidenziano vantaggi per Random Forest rispetto agli altri.",
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

    # Datos MCC reales por modelo
    mcc_names = ["Random Forest", "CNN", "Regresión Logistica"]
    mcc_vals = [0.9379, 0.6860, 0.5491]

    fig, ax = plt.subplots()
    ax.bar(mcc_names, mcc_vals, color="skyblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("MCC")
    ax.set_title(t.get("mcc_chart_title", "MCC"))
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    mcc_path = "mcc_plot.png"
    with open(mcc_path, "wb") as f:
        f.write(buf.getvalue())

    # Datos de la prueba de McNemar
    comparisons = [
        "Random Forest vs CNN",
        "Random Forest vs Regresión Logistica",
        "Regresión Logistica vs CNN",
    ]
    stats = [38.0, 28.0, 146.0]
    p_vals = [0.00001, 0.00002, 0.00001]

    fig2, ax2 = plt.subplots()
    index = np.arange(len(comparisons))
    ax2.bar(index, stats, color="lightgreen")
    ax2.set_xticks(index)
    ax2.set_xticklabels(comparisons, rotation=15)
    ax2.set_ylabel("Statistic")
    ax2.set_title(t.get("mcnemar_chart_title", "McNemar"))
    for i, p in enumerate(p_vals):
        ax2.text(i, stats[i], f"p={p:.5f}", ha="center", va="bottom", fontsize=8)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="PNG")
    plt.close(fig2)
    buf2.seek(0)
    mc_path = "mcnemar_plot.png"
    with open(mc_path, "wb") as f:
        f.write(buf2.getvalue())

    # Construcción del PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, txt=t["pdf_compare_title"], ln=True, align="C")
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, txt=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=True, align="C")
    pdf.ln(5)
    pdf.image(mcc_path, x=10, w=190)
    pdf.ln(5)
    pdf.multi_cell(0, 10, t["mcc_description"])

    pdf.add_page()
    pdf.image(mc_path, x=10, w=190)
    pdf.ln(5)
    pdf.multi_cell(0, 10, t["mcnemar_description"])

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
