import streamlit as st
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

# ===== CONFIG =====
st.set_page_config(page_title="Assistente Visual Inteligente", layout="wide")

st.title("🧠 Assistente Visual Inteligente")
st.caption("Faça perguntas sobre uma imagem usando IA")

# ===== DEVICE =====
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== CARREGAR MODELO VQA =====
@st.cache_resource
def load_model():
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    ).to(device)
    return processor, model

processor, model = load_model()

# ===== AJUSTE DE PERGUNTA =====
def ajustar_pergunta(pergunta):
    pergunta_lower = pergunta.lower()

    # detecta pergunta aberta
    if "fazendo" in pergunta_lower or "doing" in pergunta_lower:
        return [
            "Is the person sitting?",
            "Is the person walking?",
            "Is the person eating?",
            "Is the person using a phone?",
            "Is the person working?"
        ]

    return [pergunta]

# ===== INTERPRETAÇÃO =====
def interpretar_resposta(pergunta, resposta):
    pergunta = pergunta.replace("Is the person", "A pessoa está").replace("?", "")

    if resposta.lower() == "yes":
        return f"Sim — provavelmente {pergunta.lower()}"
    elif resposta.lower() == "no":
        return "Não foi possível identificar essa ação"
    else:
        return resposta

# ===== FUNÇÃO VQA MELHORADA =====
def responder_pergunta(imagem, pergunta):
    perguntas = ajustar_pergunta(pergunta)

    melhor_resposta = ""
    melhor_score = 0

    for p in perguntas:
        encoding = processor(imagem, p, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        idx = logits.argmax(-1).item()
        score = probs[0][idx].item()
        resposta = model.config.id2label[idx]

        if score > melhor_score:
            melhor_score = score
            melhor_resposta = interpretar_resposta(p, resposta)

    return melhor_resposta, melhor_score

# ===== UI =====
uploaded_file = st.file_uploader("📁 Envie uma imagem", type=["jpg", "jpeg", "png"])

pergunta = st.text_input(
    "❓ Faça uma pergunta sobre a imagem",
    placeholder="Ex: O que essa pessoa está fazendo?"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Imagem carregada", use_container_width=True)

    if pergunta:
        with st.spinner("🧠 Analisando imagem..."):
            resposta, score = responder_pergunta(image, pergunta)

        with col2:
            st.markdown("### 💬 Resposta")
            st.success(resposta)

            # score em %
            score_percent = round(score * 100, 2)
            st.metric("Confiança do modelo", f"{score_percent}%")

            # interpretação
            if score_percent > 80:
                st.success("Alta confiança 🔥")
            elif score_percent > 50:
                st.info("Confiança moderada ⚖️")
            else:
                st.warning("Baixa confiança ⚠️")

            # ===== EXPLICAÇÃO =====
            st.markdown("### 🧠 Explicação")
            st.write(
                "O sistema reformulou a pergunta em múltiplas hipóteses e selecionou "
                "a resposta com maior probabilidade com base no conteúdo visual."
            )

else:
    st.info("Envie uma imagem para começar 🚀")