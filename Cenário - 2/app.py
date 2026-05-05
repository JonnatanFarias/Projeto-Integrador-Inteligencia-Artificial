import streamlit as st
import torch
from PIL import Image
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator  # <--- Nova biblioteca para tradução

# ===== CONFIGURAÇÃO DA PÁGINA =====
st.set_page_config(page_title="Chef Vision AI", layout="wide", page_icon="👨‍🍳")

# CSS para o tema Food
st.markdown("""
    <style>
    .main { background-color: #fffaf0; }
    .stImage > img { border-radius: 15px; border: 3px solid #ff6347; }
    [data-testid="stSidebar"] { background-color: #ff6347; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    h1, h2, h3 { color: #d2691e !important; }
    </style>
    """, unsafe_allow_html=True)

# ===== CONSTANTES =====
IMAGE_FOLDER = "pec/images"
MAX_IMAGES_PER_CATEGORY = 40
TOTAL_MAX_IMAGES = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# ===== TRADUTOR INTELIGENTE =====
def translate_query(text):
    """Traduz a busca do usuário para inglês para melhorar a precisão do CLIP"""
    try:
        translated = GoogleTranslator(source='pt', target='en').translate(text)
        return translated
    except:
        return text # Se falhar, retorna o original

# ===== FUNÇÃO DE CARREGAMENTO =====
def load_images_limited():
    images, filenames, categories = [], [], []
    if not os.path.exists(IMAGE_FOLDER):
        st.error(f"Pasta '{IMAGE_FOLDER}' não encontrada.")
        return images, filenames, categories
    
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    total_loaded = 0
    for category in sorted(os.listdir(IMAGE_FOLDER)):
        category_path = os.path.join(IMAGE_FOLDER, category)
        if os.path.isdir(category_path):
            cat_count = 0
            for file in os.listdir(category_path):
                if total_loaded >= TOTAL_MAX_IMAGES or cat_count >= MAX_IMAGES_PER_CATEGORY:
                    break
                if file.lower().endswith(valid_extensions):
                    try:
                        img_path = os.path.join(category_path, file)
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                        filenames.append(file)
                        categories.append(category.replace("_", " ").title())
                        cat_count += 1
                        total_loaded += 1
                    except: continue
            if total_loaded >= TOTAL_MAX_IMAGES: break
    return images, filenames, categories

with st.spinner("Carregando estoque de ingredientes..."):
    images, filenames, categories = load_images_limited()

@st.cache_data
def get_image_embeddings(_images):
    if not _images: return np.array([])
    batch_size = 16
    all_embeddings = []
    for i in range(0, len(_images), batch_size):
        batch = _images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embeddings = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("# 🍴 Opções")
    st.metric("Total de Fotos", len(images))
    num_results = st.slider("Sugestões", 1, 6, 3)

# ===== BUSCA =====
st.title("👨‍🍳 Chef Vision AI")
query_pt = st.text_input("🥙 O que vamos preparar?", placeholder="Ex: Um hambúrguer suculento, salada fresca...")

if query_pt and images:
    with st.spinner("Traduzindo e buscando sabores..."):
        # 1. TRADUÇÃO MÁGICA
        query_en = translate_query(query_pt)
        
        # Feedback visual opcional (comente se quiser esconder)
        st.caption(f"🔍 Buscando por: *'{query_en}'*")

        # 2. PROCESSO DO CLIP (Usando a query em inglês)
        inputs = processor(text=[query_en], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_outputs = model.get_text_features(**inputs)
            text_emb = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs
            text_emb = (text_emb / torch.norm(text_emb, p=2, dim=-1, keepdim=True)).cpu().numpy()

        img_embs = get_image_embeddings(images)
        similarities = cosine_similarity(text_emb, img_embs)[0]
        top_indices = similarities.argsort()[-num_results:][::-1]

    st.markdown(f"### 🍽️ Pratos Recomendados")
    cols = st.columns(num_results)
    for idx, col_idx in enumerate(top_indices):
        with cols[idx]:
            st.markdown(f"**{categories[col_idx]}**")
            st.image(images[col_idx], use_container_width=True)
            st.caption(f"Afinidade: {similarities[col_idx]:.2%}")

# ===== GALERIA =====
st.divider()
with st.expander("📖 Ver Cardápio Completo"):
    if images:
        gal_cols = st.columns(6)
        for i, img in enumerate(images):
            gal_cols[i % 6].image(img, caption=categories[i], use_container_width=True)