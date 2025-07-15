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

# --- Language Selector ---
languages = ["English", "Español", "Français", "Deutsch", "Português", "Italiano", "日本語", "中文"]
lang = st.sidebar.selectbox("🌐 Language / Idioma / Langue / Sprache", languages)
model_options = ["CNN", "Random Forest", "Regresión Lineal"]
selected_model = st.sidebar.selectbox("🧠 Select Model", model_options)

# --- Text dictionary ---
texts = {
    "English": {
        "title": "Skin Cancer Prediction (Simulated)",
        "upload": "Upload a skin lesion image",
        "button": "Analyze Image",
        "result": "Result (simulated)",
        "confidence": "Estimated Confidence",
        "download": "Download PDF",
        "error": "❌ Error processing the image. Please make sure it's a valid file.",
        "loaded": "Loaded image",
        "chart": "Confidence Chart by Model"
    },
    "Español": {
        "title": "Predicción de Cáncer de Piel (Simulado)",
        "upload": "Sube una imagen de lesión en la piel",
        "button": "Analizar Imagen",
        "result": "Resultado (simulado)",
        "confidence": "Confianza Estimada",
        "download": "Descargar PDF",
        "error": "❌ Error al procesar la imagen. Asegúrate de que sea un archivo válido.",
        "loaded": "Imagen cargada",
        "chart": "Gráfico de Confianza por Modelo"
    },
    "Français": {
        "title": "Prédiction du cancer de la peau (simulée)",
        "upload": "Téléversez une image de lésion cutanée",
        "button": "Analyser l'image",
        "result": "Résultat (simulé)",
        "confidence": "Confiance estimée",
        "download": "Télécharger le PDF",
        "error": "❌ Erreur lors du traitement de l'image. Assurez-vous qu'il s'agit d'un fichier valide.",
        "loaded": "Image chargée",
        "chart": "Graphique de confiance par modèle"
    },
    "Deutsch": {
        "title": "Hautkrebs-Vorhersage (Simuliert)",
        "upload": "Laden Sie ein Bild der Hautläsion hoch",
        "button": "Bild analysieren",
        "result": "Ergebnis (simuliert)",
        "confidence": "Geschätzte Zuverlässigkeit",
        "download": "PDF herunterladen",
        "error": "❌ Fehler bei der Bildverarbeitung. Bitte stellen Sie sicher, dass es sich um eine gültige Datei handelt.",
        "loaded": "Geladenes Bild",
        "chart": "Vertrauensdiagramm nach Modell"
    },
    "Português": {
        "title": "Predição de Câncer de Pele (Simulada)",
        "upload": "Envie uma imagem de lesão na pele",
        "button": "Analisar Imagem",
        "result": "Resultado (simulado)",
        "confidence": "Confiança Estimada",
        "download": "Baixar PDF",
        "error": "❌ Erro ao processar a imagem. Verifique se o arquivo é válido.",
        "loaded": "Imagem carregada",
        "chart": "Gráfico de confiança por modelo"
    },
    "Italiano": {
        "title": "Predizione del Cancro della Pelle (Simulata)",
        "upload": "Carica un'immagine della lesione cutanea",
        "button": "Analizza Immagine",
        "result": "Risultato (simulato)",
        "confidence": "Fiducia Stimata",
        "download": "Scarica PDF",
        "error": "❌ Errore nell'elaborazione dell'immagine. Assicurati che sia un file valido.",
        "loaded": "Immagine caricata",
        "chart": "Grafico di fiducia per modello"
    },
    "日本語": {
        "title": "皮膚癌予測（シミュレーション）",
        "upload": "皮膚病変画像をアップロード",
        "button": "画像を分析する",
        "result": "結果（シミュレーション）",
        "confidence": "推定信頼度",
        "download": "PDFをダウンロード",
        "error": "❌ 画像の処理中にエラーが発生しました。有効なファイルか確認してください。",
        "loaded": "画像が読み込まれました",
        "chart": "モデル別信頼度グラフ"
    },
    "中文": {
        "title": "皮肤癌预测（模拟）",
        "upload": "上传皮肤病变图像",
        "button": "分析图像",
        "result": "结果（模拟）",
        "confidence": "置信度估计",
        "download": "下载PDF",
        "error": "❌ 图像处理错误。请确保是有效文件。",
        "loaded": "图像已加载",
        "chart": "按模型的置信度图"
    }
}

# --- UI ---
st.title(texts[lang]["title"])
st.markdown(f"**{texts[lang]['upload']}**")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "jfif"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption=texts[lang]["loaded"], use_column_width=True)

        if st.button(texts[lang]["button"]):
            results = {}
            for model in model_options:
                label, confidence = simulate_prediction(image, model)
                results[model] = (label, confidence)

            sel_label, sel_conf = results[selected_model]
            st.success(f"{texts[lang]['result']}: {sel_label}")
            st.info(f"{texts[lang]['confidence']}: {sel_conf:.1f}%")

            st.subheader("📊 " + texts[lang]["chart"])
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
