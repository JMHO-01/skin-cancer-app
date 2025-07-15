import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import base64
import io
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- Traducciones multilingües ---
translations = {
    "Español": {
        "title": "Predicción de Cáncer de Piel (Simulado)",
        "upload": "Sube una imagen de lesión en la piel",
        "button": "Analizar Imagen",
        "result": "Resultado (simulado)",
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
        "pdf_image_label": "Imagen analizada"
    },
    "English": {
        "title": "Skin Cancer Prediction (Simulated)",
        "upload": "Upload a skin lesion image",
        "button": "Analyze Image",
        "result": "Result (simulated)",
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
        "pdf_image_label": "Analyzed Image"
    },
    "Français": {
        "title": "Prédiction du cancer de la peau (simulée)",
        "upload": "Téléversez une image de lésion cutanée",
        "button": "Analyser l'image",
        "result": "Résultat (simulé)",
        "confidence": "Confiance estimée",
        "download": "Télécharger le PDF",
        "error": "❌ Erreur de traitement de l’image. Assurez-vous qu’il s’agisse d’un fichier valide.",
        "chart": "Graphique de confiance par modèle",
        "recommendation_malignant": "Nous vous recommandons de consulter un dermatologue pour une évaluation.",
        "recommendation_benign": "Aucun signe inquiétant détecté, mais des contrôles réguliers sont recommandés.",
        "pdf_title": "Résultat de l'analyse du cancer de la peau",
        "pdf_result": "Résultat",
        "pdf_confidence": "Confiance estimée",
        "pdf_recommendation": "Recommandation",
        "pdf_image_label": "Image analysée"
    },
    "Deutsch": {
        "title": "Hautkrebs-Vorhersage (Simuliert)",
        "upload": "Laden Sie ein Bild der Hautläsion hoch",
        "button": "Bild analysieren",
        "result": "Ergebnis (simuliert)",
        "confidence": "Geschätzte Zuverlässigkeit",
        "download": "PDF herunterladen",
        "error": "❌ Fehler beim Verarbeiten des Bildes. Bitte stellen Sie sicher, dass es sich um eine gültige Datei handelt.",
        "chart": "Vertrauensdiagramm nach Modell",
        "recommendation_malignant": "Wir empfehlen, einen Dermatologen für eine professionelle Bewertung aufzusuchen.",
        "recommendation_benign": "Keine alarmierenden Anzeichen, aber regelmäßige Kontrollen sind ratsam.",
        "pdf_title": "Ergebnis der Hautkrebsanalyse",
        "pdf_result": "Ergebnis",
        "pdf_confidence": "Geschätzte Zuverlässigkeit",
        "pdf_recommendation": "Empfehlung",
        "pdf_image_label": "Analysiertes Bild"
    },
    "Português": {
        "title": "Predição de Câncer de Pele (Simulado)",
        "upload": "Envie uma imagem de lesão na pele",
        "button": "Analisar Imagem",
        "result": "Resultado (simulado)",
        "confidence": "Confiança Estimada",
        "download": "Baixar PDF",
        "error": "❌ Erro ao processar a imagem. Certifique-se de que é um arquivo válido.",
        "chart": "Gráfico de Confiança por Modelo",
        "recommendation_malignant": "Recomenda-se procurar um dermatologista para uma avaliação profissional.",
        "recommendation_benign": "Sem sinais alarmantes, mas exames regulares são recomendados.",
        "pdf_title": "Resultado da Análise de Câncer de Pele",
        "pdf_result": "Resultado",
        "pdf_confidence": "Confiança estimada",
        "pdf_recommendation": "Recomendação",
        "pdf_image_label": "Imagem analisada"
    },
    "Italiano": {
        "title": "Previsione del Cancro della Pelle (Simulato)",
        "upload": "Carica un'immagine della lesione cutanea",
        "button": "Analizza Immagine",
        "result": "Risultato (simulato)",
        "confidence": "Affidabilità stimata",
        "download": "Scarica PDF",
        "error": "❌ Errore durante l'elaborazione dell'immagine. Assicurati che il file sia valido.",
        "chart": "Grafico di Affidabilità per Modello",
        "recommendation_malignant": "Si consiglia di consultare un dermatologo per una valutazione professionale.",
        "recommendation_benign": "Nessun segno allarmante rilevato, ma controlli regolari sono consigliati.",
        "pdf_title": "Risultato dell'Analisi del Cancro della Pelle",
        "pdf_result": "Risultato",
        "pdf_confidence": "Affidabilità stimata",
        "pdf_recommendation": "Raccomandazione",
        "pdf_image_label": "Immagine analizzata"
    },
    "日本語": {
        "title": "皮膚がん予測（シミュレーション）",
        "upload": "皮膚病変の画像をアップロードしてください",
        "button": "画像を解析する",
        "result": "結果（シミュレート）",
        "confidence": "推定信頼度",
        "download": "PDFをダウンロード",
        "error": "❌ 画像の処理中にエラーが発生しました。正しいファイルであることを確認してください。",
        "chart": "モデル別信頼度チャート",
        "recommendation_malignant": "専門的な評価のために皮膚科医の診察を受けてください。",
        "recommendation_benign": "異常は見られませんが、定期的な検診をお勧めします。",
        "pdf_title": "皮膚がん解析結果",
        "pdf_result": "結果",
        "pdf_confidence": "推定信頼度",
        "pdf_recommendation": "推奨事項",
        "pdf_image_label": "解析された画像"
    },
    "中文": {
        "title": "皮肤癌预测（模拟）",
        "upload": "上传皮肤病变图像",
        "button": "分析图像",
        "result": "结果（模拟）",
        "confidence": "估计置信度",
        "download": "下载PDF",
        "error": "❌ 图像处理出错。请确保文件有效。",
        "chart": "按模型的置信图",
        "recommendation_malignant": "建议就诊皮肤科医生进行专业评估。",
        "recommendation_benign": "未检测到异常迹象，但建议定期检查。",
        "pdf_title": "皮肤癌分析结果",
        "pdf_result": "结果",
        "pdf_confidence": "估计置信度",
        "pdf_recommendation": "建议",
        "pdf_image_label": "分析图像"
    }
}

# --- Simulated prediction ---
def simulate_prediction(image, model_name):
    np.random.seed(len(model_name) + len(image.getbands()))
    confidence = np.random.uniform(50, 100)
    label = "Malignant" if confidence > 70 else "Benign"
    return label, confidence

# --- PDF generation ---
def generate_pdf(result, confidence, language, image):
    t = translations[language]
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=t["pdf_title"], ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"{t['pdf_result']}: {result}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_confidence']}: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_recommendation']}: {t['recommendation_malignant' if result == 'Malignant' else 'recommendation_benign']}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=t["pdf_image_label"], ln=True)
    pdf.image(image_path, x=10, y=None, w=100)
    return pdf.output(dest='S').encode('latin1')

# --- UI ---
lang = st.sidebar.selectbox("🌐 Language / Idioma / Langue", list(translations.keys()))
t = translations[lang]
model_options = ["CNN", "Random Forest", "Regresión Lineal"]
selected_model = st.sidebar.selectbox("🧠 Model", model_options)

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
                label, confidence = simulate_prediction(image, model)
                results[model] = (label, confidence)

            sel_label, sel_conf = results[selected_model]
            st.success(f"{t['result']}: {sel_label}")
            st.info(f"{t['confidence']}: {sel_conf:.1f}%")

            st.subheader("📊 " + t["chart"])
            fig, ax = plt.subplots()
            ax.bar(results.keys(), [c for _, c in results.values()], color=["green", "blue", "orange"])
            ax.set_ylabel('%')
            ax.set_ylim(0, 100)
            st.pyplot(fig)

            pdf_bytes = generate_pdf(sel_label, sel_conf, lang, image)
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="prediction_report.pdf">{t["download"]}</a>'
            st.markdown(href, unsafe_allow_html=True)

    except UnidentifiedImageError:
        st.error(t["error"])
    except Exception as e:
        st.error(f"{t['error']} ({str(e)})")
