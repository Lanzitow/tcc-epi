import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Configuração da página
st.set_page_config(page_title="TCC - Deteção de EPIs", layout="centered")
st.title("🏗️ Auditoria de Segurança na Construção Civil")
st.write("Faça o upload de uma imagem do canteiro de obras para verificar automaticamente o uso de Capacetes, Coletes e Botas de segurança.")

# Carrega o modelo de forma otimizada para não travar a memória do servidor
@st.cache_resource
def carregar_modelo():
    return YOLO('best.pt')

model = carregar_modelo()

# Barra lateral para configurações
st.sidebar.header("Configurações do Modelo")
confianca = st.sidebar.slider("Limiar de Confiança (Confidence)", min_value=0.1, max_value=1.0, value=0.50, step=0.05)

# Upload da imagem
arquivo_imagem = st.file_uploader("Selecione uma fotografia...", type=['jpg', 'jpeg', 'png'])

if arquivo_imagem is not None:
    # Lê a imagem
    imagem_pil = Image.open(arquivo_imagem)
    st.write("Processando imagem...")
    
    # Converte para o formato que o YOLO (OpenCV) entende
    imagem_numpy = np.array(imagem_pil)
    imagem_cv2 = cv2.cvtColor(imagem_numpy, cv2.COLOR_RGB2BGR)
    
    # Faz a predição usando o modelo e o limiar escolhido
    resultados = model(imagem_cv2, conf=confianca)
    
    # Desenha as caixas (bounding boxes) e converte de volta para mostrar na tela
    imagem_anotada = resultados[0].plot()
    imagem_final = cv2.cvtColor(imagem_anotada, cv2.COLOR_BGR2RGB)
    
    # Exibe o resultado final na tela
    st.image(imagem_final, caption="Resultado da Deteção por Inteligência Artificial", use_container_width=True)
