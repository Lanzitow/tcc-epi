import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 1. Configuração da página (Mais moderna e larga)
st.set_page_config(page_title="TCC - Deteção de EPIs", page_icon="👷‍♂️", layout="wide")

# 2. Cabeçalho com estilo e cores
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>👷‍♂️ Sistema de Auditoria de Segurança</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #5D6D7E;'>Analise automaticamente o uso de Equipamentos de Proteção Individual em obras.</p>", unsafe_allow_html=True)
st.divider()

# Carrega o modelo de forma otimizada
@st.cache_resource
def carregar_modelo():
    return YOLO('best.pt')

model = carregar_modelo()

# 3. Layout dividido em duas colunas (Esquerda: Menus | Direita: Imagem e Resultados)
col1, col2 = st.columns([1, 3])

with col1:
    st.header("⚙️ Configurações")
    st.write("Ajuste a sensibilidade da IA:")
    confianca = st.slider("Confiança Mínima", min_value=0.1, max_value=1.0, value=0.50, step=0.05, help="Aumente para reduzir falsos alarmes.")
    
    st.write("---")
    arquivo_imagem = st.file_uploader("Carregue a foto (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

with col2:
    if arquivo_imagem is not None:
        imagem_pil = Image.open(arquivo_imagem)
        imagem_numpy = np.array(imagem_pil)
        imagem_cv2 = cv2.cvtColor(imagem_numpy, cv2.COLOR_RGB2BGR)
        
        with st.spinner("A analisar a imagem com Inteligência Artificial..."):
            # A IA faz a deteção
            resultados = model(imagem_cv2, conf=confianca)
            
            imagem_anotada = resultados[0].plot()
            imagem_final = cv2.cvtColor(imagem_anotada, cv2.COLOR_BGR2RGB)
            
            # Extrair dados para as mensagens dinâmicas
            nomes_classes = resultados[0].names
            classes_ids = resultados[0].boxes.cls.cpu().tolist() # Pega os IDs de tudo o que foi achado
            
            # Dicionário para traduzir o nome das classes para português
            traducao = {'helmet': 'Capacete', 'Vest': 'Colete', 'shoes': 'Botas de segurança'}
            
            # Descobrir quais EPIs únicos estão na foto
            epis_detetados = set()
            for cid in classes_ids:
                nome_original = nomes_classes[int(cid)]
                epis_detetados.add(traducao.get(nome_original, nome_original))
            
            # Mostra a imagem grande e bonita
            st.image(imagem_final, caption="Processamento Concluído", use_container_width=True)
            
            # 4. Painel de Mensagens Interativas
            st.subheader("📊 Resultados da Análise")
            
            if len(epis_detetados) > 0:
                lista_epis = " e ".join(epis_detetados)
                st.success(f"✅ **EPIs detetados na foto:** {lista_epis}")
                
                total_objetos = len(classes_ids)
                st.info(f"ℹ️ A Inteligência Artificial encontrou **{total_objetos} equipamento(s)** no total com confiança superior a {confianca*100:.0f}%.")
            else:
                st.warning("⚠️ Nenhum EPI detetado nesta imagem. Tente diminuir a confiança ou carregue outra fotografia.")
    else:
        # Mensagem de espera antes do utilizador colocar a foto
        st.info("👈 Por favor, carregue uma imagem de um canteiro de obras no menu lateral para iniciar a auditoria.")
