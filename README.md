# 🧠 Projeto Integrador — Inteligência Artificial Multimodal

## 📌 Visão Geral

Este repositório apresenta a implementação de três cenários práticos de Inteligência Artificial, explorando o uso de modelos multimodais capazes de interpretar imagens e linguagem natural.

Os projetos foram desenvolvidos utilizando **Python + Streamlit + modelos da Hugging Face**, com foco em aplicações reais de:

- Visão Computacional  
- Processamento de Linguagem Natural (NLP)  
- Modelos Multimodais  

---

## 🚀 Cenários Desenvolvidos

### 📷 Cenário 1 — Análise Inteligente de Imagens

#### 🎯 Objetivo
Desenvolver um sistema capaz de interpretar imagens e gerar descrições automáticas.

#### 🧠 Tecnologias
- BLIP (Image Captioning)  
- Transformers (Hugging Face)  
- PyTorch  
- Streamlit  
- Deep Translator  

#### ⚙️ Funcionamento
1. Upload de imagem  
2. Processamento com modelo BLIP  
3. Geração de descrição em inglês  
4. Tradução automática para português  
5. Cálculo de confiança da resposta  

#### 💡 Diferenciais
- Tradução automática  
- Score de confiança (%)  
- Insights da imagem  

#### 📊 Exemplo

> "A dog running in a grassy field"  
> "Um cachorro correndo em um campo gramado"  
> **Confiança:** 87%

---

### 🔎 Cenário 2 — Busca Semântica Multimodal (Imagem ↔ Texto)

#### 🎯 Objetivo
Criar um sistema que busca imagens a partir de descrições textuais.

#### 🧠 Tecnologias
- CLIP (OpenAI)  
- Transformers  
- Scikit-learn (cosine similarity)  
- NumPy  
- Streamlit  

#### ⚙️ Funcionamento
1. Usuário digita descrição em português  
2. Tradução automática para inglês  
3. Geração de embedding do texto  
4. Comparação com embeddings de imagens  
5. Retorno das imagens mais relevantes  

#### 💡 Diferenciais
- Busca semântica (não apenas palavras-chave)  
- Tradução inteligente  
- Balanceamento de dataset  
- Processamento em batch  

#### 📊 Aplicações
- E-commerce  
- Recomendação de produtos  
- Busca visual inteligente  

---

### 🤖 Cenário 3 — Assistente Visual Inteligente (VQA)

#### 🎯 Objetivo
Criar um sistema capaz de responder perguntas sobre imagens.

#### 🧠 Tecnologias
- VQA (Visual Question Answering)  
- Transformers  
- PyTorch  
- Streamlit  

#### ⚙️ Funcionamento
1. Upload da imagem  
2. Usuário faz uma pergunta  
3. Modelo interpreta imagem + pergunta  
4. Geração da resposta  

#### 💡 Diferenciais
- Interação natural com o usuário  
- Integração com LLM para melhorar respostas  
- Respostas mais explicativas  

#### 📊 Exemplo

**Pergunta:**
> "O que a pessoa está fazendo?"

**Resposta:**
> "A pessoa está sentada utilizando um computador."

---

## 🧪 Tecnologias Utilizadas

- Python  
- Streamlit  
- PyTorch  
- Hugging Face Transformers  
- PIL  
- NumPy  
- Scikit-learn  
- Deep Translator  

---

## ⚙️ Instalação

```bash
pip install torch torchvision transformers streamlit pillow numpy scikit-learn deep-translator
````

---

## ▶️ Como Executar

```bash
streamlit run app.py
```

---

## 🧠 Aprendizados

Durante o desenvolvimento, foram explorados conceitos como:

* Modelos multimodais
* Embeddings vetoriais
* Similaridade semântica
* Tradução automática aplicada à IA
* Integração de modelos pré-treinados

---

## 🚀 Possíveis Melhorias

* Deploy em nuvem (Streamlit Cloud / AWS)
* Uso de FAISS para buscas mais rápidas
* Upload de imagens no cenário 2
* Integração com APIs externas
* Melhorias de interface (UX/UI)

---

## ✅ Conclusão

Os três cenários demonstram a evolução do uso de IA:

| Cenário | Capacidade          |
| ------- | ------------------- |
| 1       | Descrever imagens   |
| 2       | Buscar imagens      |
| 3       | Responder perguntas |

Juntos, formam uma base sólida para sistemas mais avançados de IA, como:

* Assistentes inteligentes
* Sistemas de recomendação
* Interfaces conversacionais com visão computacional

---

## 👨‍💻 Autor

Projeto desenvolvido como parte de estudo/prática em Inteligência Artificial.
