import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
import torch
import asyncio
import sys
import hashlib

# ===== CORREÇÃO WINDOWS =====
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ===== CONFIG =====
st.set_page_config(page_title="Análise Inteligente", layout="wide")

# ===== ESTADO =====
if "executar" not in st.session_state:
    st.session_state.executar = False

if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

if "resultado" not in st.session_state:
    st.session_state.resultado = None

# ===== HASH =====
def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# ===== MODELO =====
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# ===== TRADUÇÃO =====
def traduzir(texto):
    try:
        return GoogleTranslator(source='en', target='pt').translate(texto)
    except:
        return f"(Tradução indisponível) → {texto}"

# ===== SCORE DE CONFIANÇA =====
def calcular_confianca(output):
    try:
        scores = output.scores  # logits de cada passo
        probs = [torch.softmax(s, dim=-1).max().item() for s in scores]
        confianca = sum(probs) / len(probs)
        return round(confianca * 100, 2)
    except:
        return 0

# ===== UI =====
st.title("🧠 Análise Inteligente de Imagens")
st.caption("Geração automática de descrição com IA + tradução 🇧🇷")

uploaded_file = st.file_uploader("📁 Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    current_hash = get_image_hash(image)

    col1, col2 = st.columns([1, 0.8])

    # ===== IMAGEM =====
    with col1:
        st.image(image, caption="Imagem carregada", width=500)

    # ===== AÇÕES =====
    with col2:
        st.markdown("### ⚙️ Ações")

        gerar = st.button("🚀 Gerar descrição", use_container_width=True)
        reset = st.button("🔄 Nova análise", use_container_width=True)

        if current_hash == st.session_state.last_image_hash:
            st.info("📌 Mesma imagem detectada")

    if gerar:
        st.session_state.executar = True

    if reset:
        st.session_state.executar = False
        st.session_state.resultado = None

    # ===== PROCESSAMENTO =====
    if st.session_state.executar:

        if st.session_state.resultado and current_hash == st.session_state.last_image_hash:
            caption_en, translated, score = st.session_state.resultado

        else:
            with st.spinner("🔍 Analisando imagem..."):
                inputs = processor(image, return_tensors="pt")

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=5,
                        output_scores=True,
                        return_dict_in_generate=True
                    )

                caption_en = processor.decode(out.sequences[0], skip_special_tokens=True)

                # calcula score aqui
                score = calcular_confianca(out)

            with st.spinner("🌎 Traduzindo..."):
                translated = traduzir(caption_en)

            st.session_state.resultado = (caption_en, translated, score)
            st.session_state.last_image_hash = current_hash

        # ===== RESULTADOS =====
        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🌍 Inglês")
            st.success(caption_en)

        with c2:
            st.markdown("### 🇧🇷 Português")
            st.success(translated)

        # ===== INSIGHTS =====
        palavras = caption_en.split()

        st.markdown("### 💡 Insights")

        i1, i2, i3, i4 = st.columns(4)

        i1.metric("Palavras", len(palavras))
        i2.metric("Primeira", palavras[0] if palavras else "-")
        i3.metric("Modelo", "BLIP")
        i4.metric("Confiança", f"{score}%")

else:
    st.info("Envie uma imagem para começar 🚀")