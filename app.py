import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io

# 1. Configuração Avançada da Página
st.set_page_config(page_title="Smart Safety - IA", page_icon="🛡️", layout="wide")

# Custom CSS para deixar as métricas mais bonitas
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    padding: 5% 10% 5% 10%;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# 2. Cabeçalho Dashboard
st.markdown("<h1 style='text-align: center; color: #1F618D;'>🛡️ Smart Safety: Visão Computacional para EPIs</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: #5D6D7E;'>Auditoria automatizada de canteiros de obras alimentada por Deep Learning (YOLO)</p>", unsafe_allow_html=True)
st.divider()

# Carregamento otimizado do modelo
@st.cache_resource
def carregar_modelo():
    return YOLO('best.pt')

model = carregar_modelo()

# 3. Painel Lateral (Sidebar) mais profissional
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1942/1942854.png", width=80) 
    st.header("⚙️ Painel de Controle")
    confianca = st.slider("🔍 Limiar de Confiança (Confidence)", 0.1, 1.0, 0.50, 0.05, help="Filtra a certeza matemática da IA.")
    st.markdown("---")
    arquivo_imagem = st.file_uploader("📸 Faça upload da imagem", type=['jpg', 'jpeg', 'png'])
    st.markdown("---")
    st.caption("Desenvolvido para TCC - Ciência de Dados/IA")

# 4. Área Principal
if arquivo_imagem is not None:
    imagem_pil = Image.open(arquivo_imagem)
    imagem_numpy = np.array(imagem_pil)
    imagem_cv2 = cv2.cvtColor(imagem_numpy, cv2.COLOR_RGB2BGR)

    with st.spinner("🧠 A rede neural está a analisar a imagem em tempo real..."):
        # Inferência
        resultados = model(imagem_cv2, conf=confianca)
        imagem_anotada = resultados[0].plot()
        imagem_final = cv2.cvtColor(imagem_anotada, cv2.COLOR_BGR2RGB)

        # Extração de Dados da IA para o Dashboard
        boxes = resultados[0].boxes
        classes_ids = boxes.cls.cpu().tolist()
        confiancas = boxes.conf.cpu().tolist()
        nomes_classes = resultados[0].names

        contagem = {'helmet': 0, 'Vest': 0, 'shoes': 0}
        detalhes = []

        for cid, conf in zip(classes_ids, confiancas):
            nome = nomes_classes[int(cid)]
            if nome in contagem:
                contagem[nome] += 1
            # Traduz e guarda para a tabela
            nome_pt = nome.replace('helmet', 'Capacete').replace('Vest', 'Colete').replace('shoes', 'Botas')
            detalhes.append({"Equipamento Detetado": nome_pt, "Confiança da IA": f"{conf*100:.1f}%"})

    # 5. Cartões de Métricas (O "Wow Factor")
    st.markdown("### 📊 Relatório Instantâneo de Deteção")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("👷 Capacetes", contagem['helmet'])
    col_m2.metric("🦺 Coletes de Visibilidade", contagem['Vest'])
    col_m3.metric("🥾 Botas de Segurança", contagem['shoes'])
    st.markdown("---")

    # 6. Sistema de Abas Interativas
    tab1, tab2 = st.tabs(["👁️ Visualizador de Imagem", "📋 Tabela de Dados (Exportável)"])

    with tab1:
        st.image(imagem_final, caption="Caixas Delimitadoras traçadas pelo modelo YOLO", use_container_width=True)
        
        # Gerar botão de download da imagem processada
        img_pil_final = Image.fromarray(imagem_final)
        buf = io.BytesIO()
        img_pil_final.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 Fazer Download da Imagem Auditada",
            data=byte_im,
            file_name="auditoria_epi_ia.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    with tab2:
        if detalhes:
            st.write("Dados extraídos da inferência do modelo para integração com banco de dados:")
            df = pd.DataFrame(detalhes)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Nenhum equipamento atingiu o limiar de confiança configurado.")
            
else:
    # Ecrã Inicial quando não há foto
    st.info("👋 Bem-vindo ao sistema **Smart Safety**! Utilize o painel esquerdo para carregar uma fotografia do canteiro de obras e iniciar a auditoria algorítmica.")
